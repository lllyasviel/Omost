import os

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = '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'  # Remember to invalid this token when public repo

import lib_omost.memory_management as memory_management

import torch
import numpy as np
import gradio as gr

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


# # SDXL
#
# sdxl_name = 'SG161222/RealVisXL_V4.0'
# # sdxl_name = 'stabilityai/stable-diffusion-xl-base-1.0'
#
# tokenizer = CLIPTokenizer.from_pretrained(
#     sdxl_name, subfolder="tokenizer")
# tokenizer_2 = CLIPTokenizer.from_pretrained(
#     sdxl_name, subfolder="tokenizer_2")
# text_encoder = CLIPTextModel.from_pretrained(
#     sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
# text_encoder_2 = CLIPTextModel.from_pretrained(
#     sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
# vae = AutoencoderKL.from_pretrained(
#     sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
# unet = UNet2DConditionModel.from_pretrained(
#     sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")
#
# unet.set_attn_processor(AttnProcessor2_0())
# vae.set_attn_processor(AttnProcessor2_0())
#
# pipeline = StableDiffusionXLOmostPipeline(
#     vae=vae,
#     text_encoder=text_encoder,
#     tokenizer=tokenizer,
#     text_encoder_2=text_encoder_2,
#     tokenizer_2=tokenizer_2,
#     unet=unet,
#     scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
# )
#
# memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])
#
# # This negative prompt is suggested by RealVisXL_V4 author
# # See also https://huggingface.co/SG161222/RealVisXL_V4.0
# # Note that in A111's normalization, a full "(full sentence)" is equal to "full sentence"
# # so we can just remove SG161222's braces
#
# default_negative = 'face asymmetry, eyes asymmetry, deformed eyes, open mouth'
#
# # LLM
#
# # model_name = 'lllyasviel/omost-phi-3-mini-128k-8bits'
# llm_name = 'lllyasviel/omost-llama-3-8b-4bits'
# # model_name = 'lllyasviel/omost-dolphin-2.9-llama3-8b-4bits'
#
# llm_model = AutoModelForCausalLM.from_pretrained(
#     llm_name,
#     torch_dtype=torch.bfloat16,  # This is computation type, not load/memory type. The loading quant type is baked in config.
#     token=HF_TOKEN,
#     device_map="auto"  # This will load model to gpu with an offload system
# )
#
# llm_tokenizer = AutoTokenizer.from_pretrained(
#     llm_name,
#     token=HF_TOKEN
# )
#
# memory_management.unload_all_models(llm_model)


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


# with torch.inference_mode():
#     example_response = open('./example_outputs.md', 'rt', encoding='utf-8').read()
#     example_response = '```python\n' + example_response + '\n```'
#
#     canvas = omost_canvas.Canvas.from_bot_response(example_response)
#     canvas_outputs = canvas.process()
#
#     guidance_scale = 7.0
#     hr_fix = False
#     use_initial_latent = False
#     H, W = 1024, 1024
#
#     rng = torch.Generator(device=memory_management.gpu).manual_seed(98654)
#
#     memory_management.load_models_to_gpu([text_encoder, text_encoder_2])
#
#     positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, default_negative)
#
#     if use_initial_latent:
#         memory_management.load_models_to_gpu([vae])
#         initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
#         initial_latent_blur = 40
#         initial_latent = torch.nn.functional.avg_pool2d(
#             torch.nn.functional.pad(initial_latent, (initial_latent_blur, ) * 4, mode='reflect'),
#             kernel_size=(initial_latent_blur * 2 + 1, ) * 2, stride=(1, 1))
#         initial_latent = torch.nn.functional.interpolate(initial_latent, (H, W))
#         initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
#         initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
#     else:
#         initial_latent = torch.zeros(size=(1, 4, H // 8, W // 8), dtype=torch.float32)
#
#     memory_management.load_models_to_gpu([unet])
#
#     initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)
#
#     latents = pipeline(
#         initial_latent=initial_latent,
#         strength=1.0,
#         num_inference_steps=25,
#         batch_size=1,
#         prompt_embeds=positive_cond,
#         negative_prompt_embeds=negative_cond,
#         pooled_prompt_embeds=positive_pooler,
#         negative_pooled_prompt_embeds=negative_pooler,
#         generator=rng,
#         guidance_scale=guidance_scale,
#     ).images
#
#     memory_management.load_models_to_gpu([vae])
#     latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
#     pixels = vae.decode(latents).sample
#     B, C, H, W = pixels.shape
#     pixels = pytorch2numpy(pixels)
#
#     if hr_fix:
#         highres_scale = 1.5
#         highres_denoise = 0.35
#         highres_steps_diffusers = 10
#
#         pixels = [
#             resize_without_crop(
#                 image=p,
#                 target_width=int(round(W * highres_scale / 64.0) * 64),
#                 target_height=int(round(H * highres_scale / 64.0) * 64)
#             ) for p in pixels
#         ]
#
#         pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
#         latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
#
#         memory_management.load_models_to_gpu([unet])
#         latents = latents.to(device=unet.device, dtype=unet.dtype)
#
#         latents = pipeline(
#             initial_latent=latents,
#             strength=highres_denoise,
#             num_inference_steps=highres_steps_diffusers,
#             batch_size=1,
#             prompt_embeds=positive_cond,
#             negative_prompt_embeds=negative_cond,
#             pooled_prompt_embeds=positive_pooler,
#             negative_pooled_prompt_embeds=negative_pooler,
#             generator=rng,
#             guidance_scale=guidance_scale,
#         ).images
#
#         memory_management.load_models_to_gpu([vae])
#         latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
#         pixels = vae.decode(latents).sample
#         pixels = pytorch2numpy(pixels)
#
#     for i in range(len(pixels)):
#         image = Image.fromarray(pixels[i])
#         image.save(f"output_{i}.png")
#
#     exit(0)


def chat_fn(message: str, history: list, temperature: float, top_p: float, max_new_tokens: int) -> str:
    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]

    for user, assistant in history:
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


css = '''
code {white-space: pre-wrap !important;}
.gradio-container {max-width: none !important;}
'''

with gr.Blocks(fill_height=True, css=css) as demo:
    with gr.Row():
        with gr.Column(scale=25):
            gr.Slider(minimum=0.0,
                      maximum=2.0,
                      step=0.01,
                      value=0.6,
                      label="Temperature")
        with gr.Column(scale=75):
            ChatInterface(
                fn=chat_fn,
                chatbot=gr.Chatbot(label='Omost', scale=1, bubble_full_width=True, render=False),
                additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
                additional_inputs=[
                    gr.Slider(minimum=0.0,
                              maximum=2.0,
                              step=0.01,
                              value=0.6,
                              label="Temperature",
                              render=False),
                    gr.Slider(minimum=0.0,
                              maximum=1.0,
                              step=0.01,
                              value=0.9,
                              label="Top P",
                              render=False),
                    gr.Slider(minimum=128,
                              maximum=4096,
                              step=1,
                              value=4096,
                              label="Max New Tokens",
                              render=False),
                ],
                examples=[
                    ['generate an image of a cat on a table in a room'],
                    ['make it on fire']
                ]
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
