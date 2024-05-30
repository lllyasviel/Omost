import os

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = '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'  # Remember to invalid this token when public repo

import lib_omost.memory_management as memory_management

import torch
import numpy as np
import gradio as gr
import tempfile

gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_temp_dir, exist_ok=True)

from threading import Thread

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

Phi3PreTrainedModel._supports_sdpa = True

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
from chat_interface import ChatInterface

import lib_omost.canvas as omost_canvas


# SDXL

sdxl_name = 'SG161222/RealVisXL_V4.0'
# sdxl_name = 'stabilityai/stable-diffusion-xl-base-1.0'

tokenizer = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(
    sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
unet = UNet2DConditionModel.from_pretrained(
    sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

pipeline = StableDiffusionXLOmostPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)

memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])

# LLM

# model_name = 'lllyasviel/omost-phi-3-mini-128k-8bits'
llm_name = 'lllyasviel/omost-llama-3-8b-4bits'
# model_name = 'lllyasviel/omost-dolphin-2.9-llama3-8b-4bits'

llm_model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.bfloat16,  # This is computation type, not load/memory type. The loading quant type is baked in config.
    token=HF_TOKEN,
    device_map="auto"  # This will load model to gpu with an offload system
)

llm_tokenizer = AutoTokenizer.from_pretrained(
    llm_name,
    token=HF_TOKEN
)

memory_management.unload_all_models(llm_model)


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


def chat_fn(message: str, history: list, temperature: float, top_p: float, max_new_tokens: int) -> str:
    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]

    for user, assistant in history:
        if user is None or assistant is None:
            continue
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    conversation.append({"role": "user", "content": message})

    memory_management.load_models_to_gpu(llm_model)

    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    streamer = TextIteratorStreamer(llm_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    if temperature == 0:
        generate_kwargs['do_sample'] = False

    Thread(target=llm_model.generate, kwargs=generate_kwargs).start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        # print(outputs)
        yield "".join(outputs)


def diffusion_fn(chatbot, num_samples, seed, image_width, image_height,
                 highres_scale, steps, cfg, highres_steps, highres_denoise, n_prompt):
    with torch.inference_mode():
        default_negative = 'face asymmetry, eyes asymmetry, deformed eyes, open mouth'
        example_response = open('./example_outputs.md', 'rt', encoding='utf-8').read()
        example_response = '```python\n' + example_response + '\n```'

        canvas = omost_canvas.Canvas.from_bot_response(example_response)
        canvas_outputs = canvas.process()

        guidance_scale = 7.0
        hr_fix = False
        use_initial_latent = False
        H, W = 1024, 1024

        rng = torch.Generator(device=memory_management.gpu).manual_seed(98654)

        memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

        positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, default_negative)

        if use_initial_latent:
            memory_management.load_models_to_gpu([vae])
            initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
            initial_latent_blur = 40
            initial_latent = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(initial_latent, (initial_latent_blur, ) * 4, mode='reflect'),
                kernel_size=(initial_latent_blur * 2 + 1, ) * 2, stride=(1, 1))
            initial_latent = torch.nn.functional.interpolate(initial_latent, (H, W))
            initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
            initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
        else:
            initial_latent = torch.zeros(size=(1, 4, H // 8, W // 8), dtype=torch.float32)

        memory_management.load_models_to_gpu([unet])

        initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

        latents = pipeline(
            initial_latent=initial_latent,
            strength=1.0,
            num_inference_steps=25,
            batch_size=1,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=guidance_scale,
        ).images

        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        B, C, H, W = pixels.shape
        pixels = pytorch2numpy(pixels)

        if hr_fix:
            highres_scale = 1.5
            highres_denoise = 0.35
            highres_steps_diffusers = 10

            pixels = [
                resize_without_crop(
                    image=p,
                    target_width=int(round(W * highres_scale / 64.0) * 64),
                    target_height=int(round(H * highres_scale / 64.0) * 64)
                ) for p in pixels
            ]

            pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
            latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor

            memory_management.load_models_to_gpu([unet])
            latents = latents.to(device=unet.device, dtype=unet.dtype)

            latents = pipeline(
                initial_latent=latents,
                strength=highres_denoise,
                num_inference_steps=highres_steps_diffusers,
                batch_size=1,
                prompt_embeds=positive_cond,
                negative_prompt_embeds=negative_cond,
                pooled_prompt_embeds=positive_pooler,
                negative_pooled_prompt_embeds=negative_pooler,
                generator=rng,
                guidance_scale=guidance_scale,
            ).images

            memory_management.load_models_to_gpu([vae])
            latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
            pixels = vae.decode(latents).sample
            pixels = pytorch2numpy(pixels)

        for i in range(len(pixels)):
            image_path = os.path.join(gradio_temp_dir, f"output_{i}.png")
            image = Image.fromarray(pixels[i])
            image.save(image_path)
            chatbot = chatbot + [(None, (image_path, 'image'))]
    return chatbot


css = '''
code {white-space: pre-wrap !important;}
.gradio-container {max-width: none !important;}
.outer_parent {flex: 1;}
.inner_parent {flex: 1;}
footer {display: none !important; visibility: hidden !important;}
'''

with gr.Blocks(fill_height=True, css=css) as demo:
    with gr.Row(elem_classes='outer_parent'):
        with gr.Column(scale=25):
            with gr.Row():
                retry_btn = gr.Button("🔄 Retry", variant="secondary", size="sm", min_width=60)
                undo_btn = gr.Button("↩️ Undo", variant="secondary", size="sm", min_width=60)
                clear_btn = gr.Button("⭐️ New Chat", variant="secondary", size="sm", min_width=60)
            with gr.Accordion(open=True, label='Language Model'):
                with gr.Group():
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            step=0.01,
                            value=0.6,
                            label="Temperature")
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.9,
                            label="Top P")
                    max_new_tokens = gr.Slider(
                        minimum=128,
                        maximum=4096,
                        step=1,
                        value=4096,
                        label="Max New Tokens")
            with gr.Accordion(open=True, label='Image Diffusion Model'):
                with gr.Group():
                    with gr.Row():
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=2, step=1)
                        seed = gr.Number(label="Seed", value=12345, precision=0)
                    with gr.Row():
                        image_width = gr.Slider(label="Image Width", minimum=256, maximum=2048, value=896, step=64)
                        image_height = gr.Slider(label="Image Height", minimum=256, maximum=2048, value=1152, step=64)
                    highres_scale = gr.Slider(label="Highres Fix Scale (1.0 is disabled)", minimum=1.0, maximum=2.0, value=1.0, step=0.01)

            with gr.Accordion(open=False, label='Advanced'):
                steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=5.0, step=0.01)
                highres_steps = gr.Slider(label="Highres Fix Steps", minimum=1, maximum=100, value=20, step=1)
                highres_denoise = gr.Slider(label="Highres Fix Denoise", minimum=0.1, maximum=1.0, value=0.4, step=0.01)
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')

            render_button = gr.Button("Render the Image!", size='lg', variant="primary")

            examples = gr.Dataset(
                samples=[
                    ['generate an image of a cat on a table in a room'],
                    ['make it on fire']
                ],
                components=[gr.Textbox(visible=False)],
                label='Quick Prompts'
            )
        with gr.Column(scale=75, elem_classes='inner_parent'):
            chatInterface = ChatInterface(
                fn=chat_fn,
                chatbot_title='Omost',
                retry_btn=retry_btn,
                undo_btn=undo_btn,
                clear_btn=clear_btn,
                additional_inputs=[temperature, top_p, max_new_tokens],
                examples=examples
            )

    render_button.click(fn=diffusion_fn, inputs=[
        chatInterface.chatbot,
        num_samples, seed, image_width, image_height, highres_scale,
        steps, cfg, highres_steps, highres_denoise, n_prompt
    ], outputs=[chatInterface.chatbot])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
