# Save this as e.g., gradio_app_direct_render.py
import os

# --- Standard Omost Setup (Keep these) ---
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = None # Set if needed for private models, but RealVisXL is public

import lib_omost.memory_management as memory_management
import uuid
import torch
import numpy as np
import gradio as gr
import tempfile
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline # IMPORTANT: Use Omost's pipeline
import lib_omost.canvas as omost_canvas

# --- Setup Directories ---
gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_temp_dir, exist_ok=True)

# --- Load SDXL Components (Keep these) ---
sdxl_name = 'SG161222/RealVisXL_V4.0'
# sdxl_name = 'stabilityai/stable-diffusion-xl-base-1.0' # Alternative base model

print("Loading SDXL Model Components...")
tokenizer = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16") # bfloat16 vae recommended
unet = UNet2DConditionModel.from_pretrained(sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

pipeline = StableDiffusionXLOmostPipeline( # Use the Omost pipeline class
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None, # Omost pipeline uses its own sampling logic
)
print("SDXL Components Loaded.")

# Unload models initially to save VRAM
memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])
print("Models initially unloaded from GPU.")

# --- Helper Functions (Keep these) ---
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

# --- NEW Render Function ---
@torch.inference_mode()
def render_from_code(
    omost_code,  # Input code string
    num_samples, seed, image_width, image_height,
    highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt
):
    print("Received request to render from code.")
    canvas_outputs = None
    initial_latent_safe = None # For error handling display

    # 1. Parse the input code string
    try:
        print("Parsing Omost code...")
        # Make sure Canvas class is available in the execution scope
        local_vars = {'Canvas': omost_canvas.Canvas, 'np': np} # np is used in canvas.process
        exec(omost_code, {'Canvas': omost_canvas.Canvas, 'np': np}, local_vars)
        canvas_instance = local_vars.get('canvas')

        if not isinstance(canvas_instance, omost_canvas.Canvas):
            raise ValueError("Generated code did not produce a valid Canvas instance.")

        print("Processing canvas instance...")
        canvas_outputs = canvas_instance.process()
        initial_latent_safe = canvas_outputs.get('initial_latent') # Keep a copy for potential display
        print("Canvas processed successfully.")

    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error parsing or processing Omost code: {e}")
        print(f"Code that failed:\n{omost_code}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Return a placeholder or error message - returning the initial latent if available can be helpful debug
        if initial_latent_safe is not None:
             error_img = Image.fromarray(initial_latent_safe).convert("RGB")
             error_img.save(os.path.join(gradio_temp_dir,"error_parsing.png"))
             gr.Warning(f"Error processing code: {e}. Displaying coarse layout.")
             return [os.path.join(gradio_temp_dir,"error_parsing.png")]
        else:
             gr.Error(f"Error processing code: {e}. Could not generate layout.")
             return None # Return None or an empty list for the gallery

    # 2. Run Diffusion (if parsing succeeded)
    if canvas_outputs:
        print("Starting diffusion process...")
        try:
            use_initial_latent = False # Set to True to use the coarse color map as initial latent
            eps = 0.05

            image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64

            rng = torch.Generator(device=memory_management.gpu).manual_seed(int(seed))

            print("Loading text encoders...")
            memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

            positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)
            print("Text prompts encoded.")

            initial_latent_source = None
            if use_initial_latent and 'initial_latent' in canvas_outputs:
                print("Preparing initial latent from canvas colors...")
                memory_management.load_models_to_gpu([vae])
                initial_latent_vis = canvas_outputs['initial_latent']
                initial_latent_source = torch.from_numpy(initial_latent_vis)[None].movedim(-1, 1) / 127.5 - 1.0
                initial_latent_blur = 40 # As used in original demo readme example
                initial_latent_source = torch.nn.functional.avg_pool2d(
                    torch.nn.functional.pad(initial_latent_source, (initial_latent_blur,) * 4, mode='reflect'),
                    kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
                initial_latent_source = torch.nn.functional.interpolate(initial_latent_source, (image_height // 8, image_width // 8), mode='bicubic', antialias=True)
                initial_latent_source = initial_latent_source.to(dtype=vae.dtype, device=vae.device)
                initial_latent_source = vae.encode(initial_latent_source).latent_dist.sample(generator=rng) * vae.config.scaling_factor # Use sample for variation
                initial_latent_source = initial_latent_source.repeat(num_samples, 1, 1, 1) # Repeat for batch size
            else:
                print("Generating initial latent noise...")
                initial_latent_source = torch.randn(
                    size=(num_samples, 4, image_height // 8, image_width // 8),
                    dtype=unet.dtype if unet.dtype is not None else torch.float32, # Use unet dtype if available
                    device=memory_management.gpu,
                    generator=rng
                )

            print("Loading UNet...")
            memory_management.load_models_to_gpu([unet])
            initial_latent_source = initial_latent_source.to(dtype=unet.dtype, device=unet.device)

            print(f"Running base diffusion pass ({steps} steps)...")
            latents = pipeline(
                initial_latent=initial_latent_source, # Use the prepared initial latent
                strength=1.0, # Full strength for txt2img equivalent
                num_inference_steps=int(steps),
                batch_size=num_samples,
                prompt_embeds=positive_cond,
                negative_prompt_embeds=negative_cond,
                pooled_prompt_embeds=positive_pooler,
                negative_pooled_prompt_embeds=negative_pooler,
                generator=rng,
                guidance_scale=float(cfg),
            ).images # The Omost pipeline directly outputs latents here
            print("Base pass finished.")

            print("Loading VAE...")
            memory_management.load_models_to_gpu([vae])
            # Latents need scaling before VAE decode
            latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
            pixels = vae.decode(latents).sample
            B, C, H, W = pixels.shape
            pixels = pytorch2numpy(pixels)
            print("VAE Decode finished.")

            # High-Resolution Fix Pass
            if highres_scale > 1.0 + eps:
                print(f"Starting High-Resolution Fix (Scale: {highres_scale}, Steps: {highres_steps}, Denoise: {highres_denoise})...")
                target_h = int(round(H * highres_scale / 64.0) * 64)
                target_w = int(round(W * highres_scale / 64.0) * 64)
                print(f"Resizing images to {target_w}x{target_h}...")
                pixels = [
                    resize_without_crop(image=p, target_width=target_w, target_height=target_h)
                    for p in pixels
                ]

                pixels_tensor = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
                print("Re-encoding with VAE...")
                latents_hr = vae.encode(pixels_tensor).latent_dist.sample(generator=rng) * vae.config.scaling_factor # Use sample for variation

                print("Loading UNet for HR pass...")
                memory_management.load_models_to_gpu([unet])
                latents_hr = latents_hr.to(device=unet.device, dtype=unet.dtype)

                # Important: Need to re-encode prompts for HR pass as context might change?
                # Omost pipeline might implicitly handle this, but let's assume original encodings are okay
                # If issues arise, re-running `all_conds_from_canvas` might be needed.

                print(f"Running HR diffusion pass ({highres_steps} steps)...")
                latents_hr = pipeline(
                    initial_latent=latents_hr, # Use HR latents
                    strength=highres_denoise, # Use denoising strength
                    num_inference_steps=int(highres_steps),
                    batch_size=num_samples,
                    prompt_embeds=positive_cond, # Re-use original encodings
                    negative_prompt_embeds=negative_cond,
                    pooled_prompt_embeds=positive_pooler,
                    negative_pooled_prompt_embeds=negative_pooler,
                    generator=rng,
                    guidance_scale=float(cfg),
                ).images
                print("HR pass finished.")

                print("Loading VAE for HR decode...")
                memory_management.load_models_to_gpu([vae])
                latents_hr = latents_hr.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
                pixels = vae.decode(latents_hr).sample
                pixels = pytorch2numpy(pixels)
                print("HR VAE Decode finished.")

            # Save images and return paths
            output_image_paths = []
            print(f"Saving {len(pixels)} generated image(s)...")
            for i, p_array in enumerate(pixels):
                unique_hex = uuid.uuid4().hex
                image_path = os.path.join(gradio_temp_dir, f"{unique_hex}_{i}.png")
                image = Image.fromarray(p_array)
                image.save(image_path)
                output_image_paths.append(image_path)
                print(f"Saved image to {image_path}")

            return output_image_paths

        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Error during diffusion process: {e}")
            import traceback
            traceback.print_exc()
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            gr.Error(f"Error during image generation: {e}")
            return None # Return None or empty list for gallery
        finally:
            # Unload models after generation
            print("Unloading diffusion models from GPU...")
            memory_management.unload_all_models()

    else:
        # This case should be handled by the parsing error block, but just in case
        print("Canvas outputs were None, skipping diffusion.")
        return None


# --- Gradio UI Definition ---
css = '''
.gradio-container {max-width: none !important;}
footer {display: none !important; visibility: hidden !important;}
'''

from gradio.themes.utils import colors

with gr.Blocks(
        fill_height=True, css=css,
        theme=gr.themes.Default(primary_hue=colors.blue, secondary_hue=colors.cyan, neutral_hue=colors.gray)
) as demo:
    gr.Markdown("# Omost Direct Renderer\nPaste your Omost Python code below and click Render.")
    with gr.Row(equal_height=False):
        # --- Left Column: Settings and Code Input ---
        with gr.Column(scale=25, min_width=400):
            seed = gr.Number(label="Random Seed", value=12345, precision=0)

            # --- Code Input Area ---
            omost_code_input = gr.Code(
                label="Omost Python Code",
                language="python",
                lines=20, # Adjust height as needed
            )
            # --- Render Button ---
            render_button = gr.Button("Render the Image!", size='lg', variant="primary")

            # --- Diffusion Settings ---
            with gr.Accordion(open=True, label='Image Diffusion Model Settings'):
                with gr.Group():
                    with gr.Row():
                        image_width = gr.Slider(label="Image Width", minimum=256, maximum=2048, value=896, step=64)
                        image_height = gr.Slider(label="Image Height", minimum=256, maximum=2048, value=1152, step=64)
                    with gr.Row():
                        num_samples = gr.Slider(label="Image Number", minimum=1, maximum=12, value=1, step=1)
                        steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)

            with gr.Accordion(open=False, label='Advanced Settings'):
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=5.0, step=0.01)
                highres_scale = gr.Slider(label="HR-fix Scale (\"1\" disables)", minimum=1.0, maximum=2.0, value=1.0, step=0.01)
                highres_steps = gr.Slider(label="Highres Fix Steps", minimum=1, maximum=100, value=20, step=1)
                highres_denoise = gr.Slider(label="Highres Fix Denoise", minimum=0.1, maximum=1.0, value=0.4, step=0.01)
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality, ugly')

        # --- Right Column: Image Output ---
        with gr.Column(scale=75):
            output_gallery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery",
                columns=2, # Adjust columns as needed
                height="auto"
            )
            # You could add a placeholder text here if needed
            # gr.Markdown("Generated images will appear here.")

    # --- Button Click Action ---
    render_button.click(
        fn=render_from_code,
        inputs=[
            omost_code_input,
            num_samples, seed, image_width, image_height,
            highres_scale, steps, cfg, highres_steps, highres_denoise, n_prompt
        ],
        outputs=[output_gallery]
    )

if __name__ == "__main__":
    # Ensure required directories from Omost exist if lib_omost expects them
    # (In this case, only hf_download is explicitly mentioned)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'hf_download'), exist_ok=True)

    print("Starting Gradio App...")
    demo.queue().launch(share = True, inbrowser=True, server_name='0.0.0.0')
