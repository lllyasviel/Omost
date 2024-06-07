import os

#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #国内IP使用镜像站下载模型，不想用就注释掉
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'models/hf_download')
HF_TOKEN = None

import lib_omost.memory_management as memory_management
import uuid

import torch
import numpy as np
import gradio as gr
import tempfile

import datetime
import time

gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_temp_dir, exist_ok=True)
outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

from threading import Thread

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

Phi3PreTrainedModel._supports_sdpa = True

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLImg2ImgPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
from chat_interface import ChatInterface
from transformers.generation.stopping_criteria import StoppingCriteriaList

import lib_omost.canvas as omost_canvas

llm_models_list=[
    'lllyasviel/omost-llama-3-8b-4bits',
    'lllyasviel/omost-dolphin-2.9-llama3-8b-4bits',
    'lllyasviel/omost-phi-3-mini-128k-8bits',
    'lllyasviel/omost-llama-3-8b',
    'lllyasviel/omost-dolphin-2.9-llama3-8b',
    'lllyasviel/omost-phi-3-mini-128k'
]
# LLM default model
llm_name = 'lllyasviel/omost-llama-3-8b-4bits'

models_dir = os.path.join(os.getcwd(), 'models/checkpoints')

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"models/checkpoints folder not found, creating one at {models_dir}")   

# sdxl_name default model
sdxl_name = 'RealVisXL_V4.0'

def get_image_diffusion_models_list(models_dir):
    return [os.path.splitext(os.path.basename(file))[0] for file in os.listdir(models_dir) if file.endswith('.safetensors')]

image_diffusion_models_list = get_image_diffusion_models_list(models_dir)

def refresh_models_list():
    image_diffusion_models_list = get_image_diffusion_models_list(models_dir)
    return gr.update(choices=image_diffusion_models_list)

tokenizer = None
tokenizer_2 = None
text_encoder = None
text_encoder_2 = None
vae = None
unet = None
pipeline = None

def load_model(models_dir, image_diffusion_model_select):
    global tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae, unet, pipeline

    model_path = os.path.join(models_dir, image_diffusion_model_select + '.safetensors')
    
    if not os.path.isfile(model_path):
        print(f"{image_diffusion_model_select}.safetensors not found in {models_dir} .")
        print(f"Please download the model file from https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors to {models_dir}")
        print(f"Next, it will switch to the Hugging Face directory 'SG161222/RealVisXL_V4.0' to download and run.")
        hf_repo_id = 'SG161222/RealVisXL_V4.0'

        tokenizer = CLIPTokenizer.from_pretrained(
            hf_repo_id, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            hf_repo_id, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(
            hf_repo_id, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
        text_encoder_2 = CLIPTextModel.from_pretrained(
            hf_repo_id, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
        vae = AutoencoderKL.from_pretrained(
            hf_repo_id, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
        unet = UNet2DConditionModel.from_pretrained(
            hf_repo_id, subfolder="unet", torch_dtype=torch.float16, variant="fp16")
    else:
        # float32 is preferred for GPUs with 12GB or more of VRAM?
        model_pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(model_path, torch_dtype=torch.float16, variant="fp16")

        tokenizer = model_pipeline.tokenizer
        tokenizer_2 = model_pipeline.tokenizer_2
        text_encoder = model_pipeline.text_encoder
        text_encoder_2 = model_pipeline.text_encoder_2
        text_encoder_2 = CLIPTextModel(config=text_encoder_2.config)
        vae = model_pipeline.vae
        unet = model_pipeline.unet
    
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
    return pipeline

def process_seed(seed_string):
    try:
        seed = int(seed_string)
    except ValueError:
        raise ValueError(f"The seed string '{seed_string}' is not a valid integer.")
    if seed == -1:
        seed = np.random.randint(0, 2**31 - 1) #int32 max value
    elif not (0 <= seed <= 2**31 - 1):
        raise ValueError(f"The seed value '{seed}' is out of the valid range for int32 [0, {2**31 - 1}].")    
    return seed

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

@torch.inference_mode()
def chat_fn(message: str, history: list, seed:int, temperature: float, top_p: float, max_new_tokens: int, llm_model_select: int) -> str:          

    print(f'Loading LLM model: {llm_model_select}')

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_select,
        torch_dtype=torch.bfloat16,  # This is computation type, not load/memory type. The loading quant type is baked in config.
        token=HF_TOKEN,
        device_map="auto"  # This will load model to gpu with an offload system
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_model_select,
        token=HF_TOKEN
    )

    memory_management.unload_all_models(llm_model)

    chat_start_time = time.perf_counter()

    seed = process_seed(seed)
    print(f"Chat seed: {seed}")
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]

    for user, assistant in history:
        if isinstance(user, str) and isinstance(assistant, str):
            if len(user) > 0 and len(assistant) > 0:
                conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    conversation.append({"role": "user", "content": message})

    memory_management.load_models_to_gpu(llm_model)

    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    streamer = TextIteratorStreamer(llm_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    def interactive_stopping_criteria(*args, **kwargs) -> bool:
        if getattr(streamer, 'user_interrupted', False):
            print('User stopped generation')
            return True
        else:
            return False

    stopping_criteria = StoppingCriteriaList([interactive_stopping_criteria])

    def interrupter():
        streamer.user_interrupted = True
        return

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
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
        yield "".join(outputs), interrupter

    chat_time = time.perf_counter() - chat_start_time
    print(f'Chat total time: {chat_time:.2f} seconds')
    print('Chat finished')

    return


@torch.inference_mode()
def post_chat(history):
    canvas_outputs = None

    try:
        if history:
            history = [(user, assistant) for user, assistant in history if isinstance(user, str) and isinstance(assistant, str)]
            last_assistant = history[-1][1] if len(history) > 0 else None
            canvas = omost_canvas.Canvas.from_bot_response(last_assistant)
            canvas_outputs = canvas.process()
    except Exception as e:
        print('Last assistant response is not valid canvas:', e)

    return canvas_outputs, gr.update(visible=canvas_outputs is not None), gr.update(interactive=len(history) > 0)


@torch.inference_mode()
def diffusion_fn(chatbot, canvas_outputs, num_samples, seed, image_width, image_height,
                 highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt,image_diffusion_model_select):

    diffusion_start_time = time.perf_counter()

    use_initial_latent = False
    eps = 0.05 
  
    print(f'Loading image diffusion model: {image_diffusion_model_select}')

    pipeline = load_model(models_dir, image_diffusion_model_select)

    memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])

    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64

    seed = process_seed(seed)
    print(f"Image seed: {seed}")

    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

    positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)

    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
            kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

    memory_management.load_models_to_gpu([unet])

    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=int(steps),
        batch_size=num_samples,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=float(cfg),
    ).images

    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    B, C, H, W = pixels.shape
    pixels = pytorch2numpy(pixels)

    if highres_scale > 1.0 + eps:
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
            num_inference_steps=highres_steps,
            batch_size=num_samples,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=float(cfg),
        ).images

        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)

    for i in range(len(pixels)):
        unique_hex = uuid.uuid4().hex
        #image_path = os.path.join(gradio_temp_dir, f"{unique_hex}_{i}.png")   
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join(outputs_dir, f"{time_string}_{i}_{unique_hex}.png")           
        print(f'Image saved at: {image_path}')
        image = Image.fromarray(pixels[i])
        image.save(image_path)
        chatbot = chatbot + [(None, (image_path, 'image'))]

    diffusion_time = time.perf_counter() - diffusion_start_time
    print(f'Image render total time: {diffusion_time:.2f} seconds')

    return chatbot


css = '''
code {white-space: pre-wrap !important;}
.gradio-container {max-width: none !important;}
.outer_parent {flex: 1;}
.inner_parent {flex: 1;}
footer {display: none !important; visibility: hidden !important;}
.translucent {display: none !important; visibility: hidden !important;}
'''

from gradio.themes.utils import colors

with gr.Blocks(
        fill_height=True, css=css,
        theme=gr.themes.Default(primary_hue=colors.blue, secondary_hue=colors.cyan, neutral_hue=colors.gray)
) as demo:
    with gr.Row(elem_classes='outer_parent'):
        with gr.Column(scale=25):
            with gr.Row():
                clear_btn = gr.Button("➕ New Chat", variant="secondary", size="sm", min_width=60)
                retry_btn = gr.Button("Retry", variant="secondary", size="sm", min_width=60, visible=False)
                undo_btn = gr.Button("✏️️ Edit Last Input", variant="secondary", size="sm", min_width=60, interactive=False)
            with gr.Tab(label='Setting'):

                seed = gr.Number(label="Random Seed", value=-1, precision=0)

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
                            image_width = gr.Slider(label="Image Width", minimum=256, maximum=2048, value=896, step=64)
                            image_height = gr.Slider(label="Image Height", minimum=256, maximum=2048, value=1152, step=64)

                        with gr.Row():
                            num_samples = gr.Slider(label="Image Number", minimum=1, maximum=12, value=1, step=1)
                            steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)

                with gr.Accordion(open=False, label='Advanced'):
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=5.0, step=0.01)
                    highres_scale = gr.Slider(label="HR-fix Scale (\"1\" is disabled)", minimum=1.0, maximum=2.0, value=1.0, step=0.01)
                    highres_steps = gr.Slider(label="Highres Fix Steps", minimum=1, maximum=100, value=20, step=1)
                    highres_denoise = gr.Slider(label="Highres Fix Denoise", minimum=0.1, maximum=1.0, value=0.4, step=0.01)
                    n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')

            with gr.Tab(label='Models'):
                llm_model_select = gr.Dropdown(label="LLM model", choices=llm_models_list, value=llm_name, interactive=True)
                llm_model_select.change(inputs=[llm_model_select], outputs=[])
                image_diffusion_model_select = gr.Dropdown(label="Image diffusion model", choices=image_diffusion_models_list, value=sdxl_name, interactive=True)
                image_diffusion_model_select.change(inputs=[image_diffusion_model_select], outputs=[])
                refresh_models_list_btn = gr.Button("🔄️ Refresh Image diffusion model list", variant="secondary", min_width=60)
                refresh_models_list_btn.click(refresh_models_list, inputs=[], outputs=[image_diffusion_model_select])

            render_button = gr.Button("Render the Image!", size='lg', variant="primary", visible=False)

            examples = gr.Dataset(
                samples=[
                    ['generate an image of the fierce battle of warriors and a dragon'],
                    ['change the dragon to a dinosaur'],
                    ['generate a half length portrait photoshooot of a man and a woman on the city street']
                ],
                components=[gr.Textbox(visible=False)],
                label='Quick Prompts'
            )
        with gr.Column(scale=75, elem_classes='inner_parent'):
            canvas_state = gr.State(None)
            chatbot = gr.Chatbot(label='Omost', scale=1, show_copy_button=True, layout="panel", render=False)
            chatInterface = ChatInterface(
                fn=chat_fn,
                post_fn=post_chat,
                post_fn_kwargs=dict(inputs=[chatbot], outputs=[canvas_state, render_button, undo_btn]),
                pre_fn=lambda: gr.update(visible=False),
                pre_fn_kwargs=dict(outputs=[render_button]),
                chatbot=chatbot,
                retry_btn=retry_btn,
                undo_btn=undo_btn,
                clear_btn=clear_btn,
                additional_inputs=[seed, temperature, top_p, max_new_tokens,llm_model_select],
                examples=examples
            )

    render_button.click(
        fn=diffusion_fn, inputs=[
            chatInterface.chatbot, canvas_state,
            num_samples, seed, image_width, image_height, highres_scale,
            steps, cfg, highres_steps, highres_denoise, n_prompt,
            image_diffusion_model_select
        ], outputs=[chatInterface.chatbot]).then(
        fn=lambda x: x, inputs=[
            chatInterface.chatbot
        ], outputs=[chatInterface.chatbot_state])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, server_name='0.0.0.0')
