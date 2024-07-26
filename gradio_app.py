import argparse
import gc
import json
import os
import sys
import tempfile
import uuid
from threading import Thread

import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLImg2ImgPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList
# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

import lib_omost.canvas as omost_canvas
import lib_omost.memory_management as memory_management
from chat_interface import ChatInterface
from lib_omost.pipeline import StableDiffusionXLOmostPipeline

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = None

gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_temp_dir, exist_ok=True)

Phi3PreTrainedModel._supports_sdpa = True

parser = argparse.ArgumentParser()
parser.add_argument("--hf_token", type=str, default=None)
parser.add_argument("--sdxl_name", type=str, default='RunDiffusion/Juggernaut-X-v10')
parser.add_argument("--llm_name", type=str, default='lllyasviel/omost-llama-3-8b-4bits')
parser.add_argument("--checkpoints_folder", type=str,
                    default=os.path.join(os.path.dirname(__file__), "models", "checkpoints"))
parser.add_argument("--lora-folder", type=str, default=os.path.join(os.path.dirname(__file__), "models", "lora"))
parser.add_argument("--llm_folder", type=str, default=os.path.join(os.path.dirname(__file__), "models", "llm"))
parser.add_argument("--outputs_folder", type=str, default=os.path.join(os.path.dirname(__file__), "outputs"))
# Add a --no-defaults flag to disable the default models
parser.add_argument("--no_defaults", action='store_true')
args = parser.parse_args()

DEFAULT_CHECKPOINTS = {
    'RunDiffusion/Juggernaut-X-v10': 'Juggernaut-X-v10',
    'SG161222/RealVisXL_V4.0': 'RealVisXL_V4.0',
    'stabilityai/stable-diffusion-xl-base-1.0': 'stable-diffusion-xl-base-1.0'
}

if args.no_defaults:
    DEFAULT_CHECKPOINTS = {}

DEFAULT_LLMS = {
    'lllyasviel/omost-llama-3-8b-4bits': 'omost-llama-3-8b-4bits',
    'lllyasviel/omost-dolphin-2.9-llama3-8b-4bits': 'omost-dolphin-2.9-llama3-8b-4bits',
    'lllyasviel/omost-phi-3-mini-128k-8bits': 'omost-phi-3-mini-128k-8bits',
    'lllyasviel/omost-llama-3-8b': 'omost-llama-3-8b',
    'lllyasviel/omost-dolphin-2.9-llama3-8b': 'omost-dolphin-2.9-llama3-8b',
    'lllyasviel/omost-phi-3-mini-128k': 'omost-phi-3-mini-128k'
}

pipeline = None
loaded_pipeline = None
selected_lora = None
llm_model = None
llm_model_name = None
llm_tokenizer = None

os.makedirs(args.outputs_folder, exist_ok=True)


def list_models(llm: bool = False, lora: bool = False):
    if llm:
        folder_path = args.llm_folder
        default_model = args.llm_name
        default_models = [(name, key) for key, name in DEFAULT_LLMS.items()]
    elif lora:
        folder_path = args.lora_folder
        default_model = None
        default_models = []
    else:
        folder_path = args.checkpoints_folder
        default_model = args.sdxl_name
        default_models = [(name, key) for key, name in DEFAULT_CHECKPOINTS.items()]
    models = []
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            # If we want to list only files with a specific extension
            if not llm:
                for file in files:
                    if file.endswith(".safetensors"):
                        full_path = os.path.join(root, file)
                        models.append((file.replace(".safetensors", ""), full_path))
            else:
                # Append dirs
                for model_dir in dirs:
                    full_path = os.path.join(root, model_dir)
                    models.append((model_dir, full_path))
    if llm:
        if llm_model and llm_model_name and llm_model_name in [name for key, name in models]:
            default_model = llm_model_name
    elif lora:
        default_model = ""
    else:
        if pipeline and loaded_pipeline and loaded_pipeline in [name for key, name in models]:
            default_model = loaded_pipeline
    # Sort default_models alphabetically
    default_models = sorted(default_models, key=lambda x: x[0])
    models = sorted(models, key=lambda x: x[0])
    if default_models:
        models = default_models + models
    if lora:
        # Inject "" as the first entry
        models = [""] + models
    return models, default_model


def load_pipeline(model_path, lora):
    global pipeline, loaded_pipeline, selected_lora

    if pipeline is not None and loaded_pipeline == model_path:
        if selected_lora != lora:
            if lora is not None and lora != "" and os.path.exists(lora):
                if selected_lora != "" and selected_lora is not None:
                    pipeline.unload_lora_weights()
                pipeline.load_lora_weights(lora)
                selected_lora = lora
            else:
                pipeline.unload_lora_weights()
        return

    if pipeline:
        pipeline = None
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Loading model from {model_path}")

    if model_path.endswith('.safetensors'):
        pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(model_path, torch_dtype=torch.float16,
                                                                     variant="fp16")
        pipeline = pipeline.to(memory_management.gpu)
        tokenizer = pipeline.tokenizer
        tokenizer_2 = pipeline.tokenizer_2
        text_encoder = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        text_encoder_2 = CLIPTextModel(config=text_encoder_2.config)
        vae = pipeline.vae
        unet = pipeline.unet
    else:
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=torch.float16)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch.float16)
        text_encoder_2 = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder_2",
                                                       torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float16)

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
    if lora is not None and lora != "" and os.path.exists(lora):
        print(f"Loading Lora from {lora}")
        if selected_lora != "" and selected_lora is not None:
            pipeline.unload_lora_weights()
        pipeline.load_lora_weights(lora)
        selected_lora = lora
    else:
        pipeline.unload_lora_weights()

    pipeline = pipeline.to(torch.device('cuda'))
    loaded_pipeline = model_path


def load_llm_model(model_name, do_unload: bool = True):
    global llm_model, llm_tokenizer, llm_model_name
    if llm_model_name == model_name and llm_model:
        return
    if llm_model:
        del llm_model
    if llm_tokenizer:
        del llm_tokenizer
    torch.cuda.empty_cache()
    print(f"Loading LLM model from {model_name}")

    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        device_map="auto"
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN
    )
    llm_model_name = model_name
    # if do_unload:
    #     llm_model = llm_model.to(torch.device('cpu'))


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


def random_seed():
    if sys.maxsize > 2 ** 32:
        try:
            return np.random.randint(0, 2 ** 32 - 1)
        except:
            return np.random.randint(0, 2 ** 31 - 1)
    else:
        return np.random.randint(0, 2 ** 31 - 1)


@torch.inference_mode()
def chat_fn(message: str, history: list, seed: int, temperature: float, top_p: float, max_new_tokens: int, lora: str) -> str:
    global llm_model, llm_tokenizer, llm_model_name, pipeline

    if seed == -1:
        seed = random_seed()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]

    for user, assistant in history:
        if isinstance(user, str) and isinstance(assistant, str):
            if len(user) > 0 and len(assistant) > 0:
                conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    conversation.append({"role": "user", "content": message})
    if pipeline:
        pipeline = pipeline.to(memory_management.cpu)
    # Load the model if it is not loaded
    if not llm_model:
        load_llm_model(llm_model_name, False)

    memory_management.load_models_to_gpu([llm_model])

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
        yield "".join(outputs), interrupter

    return


@torch.inference_mode()
def post_chat(history):
    canvas_outputs = None

    try:
        if history:
            history = [(user, assistant) for user, assistant in history if
                       isinstance(user, str) and isinstance(assistant, str)]
            last_assistant = history[-1][1] if len(history) > 0 else None
            canvas = omost_canvas.Canvas.from_bot_response(last_assistant)
            canvas_outputs = canvas.process()
    except Exception as e:
        print('Last assistant response is not valid canvas:', e)

    return canvas_outputs, gr.update(visible=canvas_outputs is not None), gr.update(interactive=len(history) > 0)


@torch.inference_mode()
def diffusion_fn(chatbot, canvas_outputs, num_samples, seed, image_width, image_height,
                 highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt, model_selection,
                 lora_selection, lora_scale):
    global pipeline, llm_model, llm_tokenizer

    lora_info_dict = {}
    if lora_selection != "" and os.path.exists(lora_selection):
        lora_json = lora_selection.replace(".safetensors", ".json")
        if os.path.exists(lora_json):
            with open(lora_json, "r") as f:
                lora_info_dict = json.load(f)
    else:
        lora_scale = 0

    activation_text = lora_info_dict.get("activation text", None)

    memory_management.unload_all_models([llm_model])
    use_initial_latent = False
    eps = 0.05
    # Load the model
    load_pipeline(model_selection, lora_selection)
    if not isinstance(pipeline, StableDiffusionXLOmostPipeline):
        raise ValueError("Pipeline is not StableDiffusionXLOmostPipeline")
    vae = pipeline.vae
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    if not isinstance(vae, AutoencoderKL):
        raise ValueError("VAE is not AutoencoderKL")

    if not isinstance(unet, UNet2DConditionModel):
        raise ValueError("UNet is not UNet2DConditionModel")

    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64
    if seed == -1:
        seed = random_seed()
    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

    positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs,
                                                                                                    negative_prompt,
                                                                                                    lora_scale,
                                                                                                    activation_text)

    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
            kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        try:
            if isinstance(vae.config, dict):
                initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config['scaling_factor']
            else:
                initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
        except Exception as e:
            print('Failed to encode initial latent:', e)
            initial_latent = torch.zeros(size=(1, 4, image_height // 8, image_width // 8), dtype=torch.float32)
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

    memory_management.load_models_to_gpu([unet])

    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)
    print("Starting diffusion")
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
        cross_attention_kwargs={"scale": lora_scale},
        guidance_scale=float(cfg),
    ).images

    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    B, C, H, W = pixels.shape
    pixels = pytorch2numpy(pixels)
    print("Diffusion done, doing hires")
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
            cross_attention_kwargs={"scale": lora_scale},
            generator=rng,
            guidance_scale=float(cfg),
        ).images

        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)

    for i in range(len(pixels)):
        unique_hex = uuid.uuid4().hex
        image_path = os.path.join(args.outputs_folder, f"{unique_hex}_{i}.png")
        image = Image.fromarray(pixels[i])
        image.save(image_path)
        chatbot = chatbot + [(None, (image_path, 'image'))]

    return chatbot


def update_model_list():
    model_list, default_model = list_models(False)
    if loaded_pipeline and loaded_pipeline in [path for name, path in model_list]:
        default_model = loaded_pipeline
    options = [(model, path) for model, path in model_list]
    paths = {model: path for model, path in model_list}
    return gr.update(choices=options, value=default_model if options else ""), paths


def update_lora_list():
    model_list, default_model = list_models(lora=True)
    if loaded_pipeline and loaded_pipeline in [path for name, path in model_list]:
        default_model = loaded_pipeline
    options = [(model, path) for model, path in model_list]
    paths = {model: path for model, path in model_list}
    return gr.update(choices=options, value=default_model if options else ""), paths


def update_llm_list():
    llm_list, default_llm = list_models(True)
    if llm_model and llm_model_name and llm_model_name in [name for name, path in llm_list]:
        default_llm = llm_model_name
    options = [(os.path.basename(path), path) for name, path in llm_list]
    paths = {os.path.basename(path): path for name, path in llm_list}
    return gr.update(choices=options, value=default_llm if options else ""), paths


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
                undo_btn = gr.Button("✏️️ Edit Last Input", variant="secondary", size="sm", min_width=60,
                                     interactive=False)

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
                    llm_models, selected = list_models(True)
                    llm_model_name = selected
                    llm_select = gr.Dropdown(label="Select LLM Model", value=selected, choices=llm_models,
                                             interactive=True)
                    llm_refresh_btn = gr.Button("Refresh LLM List", variant="secondary", size="sm", min_width=60)
            with gr.Accordion(open=True, label='Image Diffusion Model'):
                with gr.Group():
                    with gr.Row():
                        image_width = gr.Slider(label="Image Width", minimum=256, maximum=2048, value=1920, step=64)
                        image_height = gr.Slider(label="Image Height", minimum=256, maximum=2048, value=1080, step=64)

                    with gr.Row():
                        with gr.Column():
                            checkpoint_list, selected = list_models(False)
                            model_select = gr.Dropdown(label="Select Model", choices=checkpoint_list, interactive=True,
                                                       value=selected)
                            model_refresh_btn = gr.Button("Refresh Model List", variant="secondary", size="sm",
                                                          min_width=60)
                        num_samples = gr.Slider(label="Image Number", minimum=1, maximum=12, value=1, step=1)
                        steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)

            with gr.Accordion(open=False, label='Advanced'):
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=5.0, step=0.01)
                highres_scale = gr.Slider(label="HR-fix Scale (\"1\" is disabled)", minimum=1.0, maximum=2.0, value=1.0,
                                          step=0.01)
                highres_steps = gr.Slider(label="Highres Fix Steps", minimum=1, maximum=100, value=20, step=1)
                highres_denoise = gr.Slider(label="Highres Fix Denoise", minimum=0.1, maximum=1.0, value=0.4, step=0.01)
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='lowres, bad anatomy, bad hands, cropped, worst quality')
                with gr.Column():
                    lora_list, lora_selected = list_models(lora=True)
                    lora_weight = gr.Slider(label="Lora Weight", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                    lora_select = gr.Dropdown(label="Select Lora (Experimental)", choices=lora_list, interactive=True,
                                              value=lora_selected)
                    lora_refresh_btn = gr.Button("Refresh Model List", variant="secondary", size="sm",
                                                 min_width=60)

            render_button = gr.Button("Render the Image!", size='lg', variant="primary", visible=False)

            examples = gr.Dataset(
                samples=[
                    ['Generate an image of several squirrels in business suits having a meeting in a park'],
                    ['Generate a photo of three goats sitting around a table playing poker.'],
                    ['Add a dog in the background']
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
                additional_inputs=[seed, temperature, top_p, max_new_tokens, lora_select],
                examples=examples
            )

    render_button.click(
        fn=diffusion_fn, inputs=[
            chatInterface.chatbot, canvas_state,
            num_samples, seed, image_width, image_height, highres_scale,
            steps, cfg, highres_steps, highres_denoise, n_prompt, model_select, lora_select, lora_weight
        ], outputs=[chatInterface.chatbot]).then(
        fn=lambda x: x, inputs=[
            chatInterface.chatbot
        ], outputs=[chatInterface.chatbot_state])

    model_refresh_btn.click(
        fn=update_model_list,
        inputs=[],
        outputs=[model_select]
    )

    lora_refresh_btn.click(
        fn=update_lora_list,
        inputs=[],
        outputs=[lora_select]
    )

    llm_refresh_btn.click(
        fn=update_llm_list,
        inputs=[],
        outputs=[llm_select]
    )

    llm_select.change(
        fn=load_llm_model,
        inputs=[llm_select],
        outputs=[]
    )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, server_name='0.0.0.0')
