# Prediction interface for Cog ⚙️
# https://cog.run/python


import os
import re
import time
import subprocess
import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, BaseModel, Path

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

Phi3PreTrainedModel._supports_sdpa = True
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
)
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
import lib_omost.canvas as omost_canvas


MODEL_URL = "https://weights.replicate.delivery/default/lllyasviel/Omost.tar"
MODEL_CACHE = "model_cache"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class ModelOutput(BaseModel):
    code: str
    image: Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        sdxl_name = "SG161222/RealVisXL_V4.0"
        tokenizer = CLIPTokenizer.from_pretrained(
            f"{MODEL_CACHE}/{sdxl_name}", subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            f"{MODEL_CACHE}/{sdxl_name}", subfolder="tokenizer_2"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            f"{MODEL_CACHE}/{sdxl_name}",
            subfolder="text_encoder",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        text_encoder_2 = CLIPTextModel.from_pretrained(
            f"{MODEL_CACHE}/{sdxl_name}",
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        self.vae = AutoencoderKL.from_pretrained(
            f"{MODEL_CACHE}/{sdxl_name}",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            variant="fp16",
        ).to("cuda")
        self.unet = UNet2DConditionModel.from_pretrained(
            f"{MODEL_CACHE}/{sdxl_name}",
            subfolder="unet",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())

        self.pipeline = StableDiffusionXLOmostPipeline(
            vae=self.vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            unet=self.unet,
            scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
        )

        llm_name = "lllyasviel/omost-llama-3-8b-4bits"
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            f"{MODEL_CACHE}/{llm_name}",  # this is the lllyasviel/omost-llama-3-8b-4bits model
            torch_dtype=torch.bfloat16,  # This is computation type, not load/memory type. The loading quant type is baked in config.
            device_map="auto",  # This will load model to gpu with an offload system
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_CACHE}/{llm_name}")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="generate an image of the fierce battle of warriors and the dragon",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="lowres, bad anatomy, bad hands, cropped, worst quality",
        ),
        image_width: int = Input(
            description="Width of output image",
            default=896,
        ),
        image_height: int = Input(
            description="Height of output image",
            default=1152,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=32, default=5
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic",
            ge=0.0,
            le=2.0,
            default=0.6,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.9,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            ge=128,
            le=4096,
            default=4096,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

        conversation = [{"role": "system", "content": omost_canvas.system_prompt}]
        conversation.append({"role": "user", "content": prompt})

        input_ids = self.llm_tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True
        ).to(self.llm_model.device)

        out = self.llm_model.generate(
            input_ids=input_ids,
            do_sample=False if temperature == 0 else True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        decoded_output = self.llm_tokenizer.decode(
            out.tolist()[0], skip_special_tokens=True, skip_prompt=True
        )

        prefix = f"system\n\n{omost_canvas.system_prompt}"
        assert decoded_output.startswith(prefix)
        response = decoded_output[len(prefix) :]

        use_initial_latent = False

        image_width, image_height = (
            int(image_width // 64) * 64,
            int(image_height // 64) * 64,
        )

        rng = torch.Generator(device="cuda").manual_seed(seed)

        matched = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        assert matched, "Response does not contain codes!"
        code_content = matched.group(1)
        assert (
            "canvas = Canvas()" in code_content
        ), "Code block must include valid canvas var!"
        local_vars = {"Canvas": omost_canvas.Canvas}
        exec(code_content, {}, local_vars)
        canvas = local_vars.get("canvas", None)
        assert isinstance(
            canvas, omost_canvas.Canvas
        ), "Code block must produce valid canvas var!"

        canvas_outputs = canvas.process()

        print("code generated, rendering the image now...")

        positive_cond, positive_pooler, negative_cond, negative_pooler = (
            self.pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)
        )
        print("done")

        if use_initial_latent:
            initial_latent = (
                torch.from_numpy(canvas_outputs["initial_latent"])[None].movedim(-1, 1)
                / 127.5
                - 1.0
            )
            initial_latent_blur = 40
            initial_latent = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(
                    initial_latent, (initial_latent_blur,) * 4, mode="reflect"
                ),
                kernel_size=(initial_latent_blur * 2 + 1,) * 2,
                stride=(1, 1),
            )
            initial_latent = torch.nn.functional.interpolate(
                initial_latent, (image_height, image_width)
            )
            initial_latent = initial_latent.to(
                dtype=self.vae.dtype, device=self.vae.device
            )
            initial_latent = (
                self.vae.encode(initial_latent).latent_dist.mode()
                * self.vae.config.scaling_factor
            )
        else:
            initial_latent = torch.zeros(
                size=(1, 4, image_height // 8, image_width // 8),
                dtype=torch.float32,
            )

        initial_latent = initial_latent.to(
            dtype=self.unet.dtype, device=self.unet.device
        )

        latents = self.pipeline(
            initial_latent=initial_latent,
            strength=1.0,
            num_inference_steps=int(num_inference_steps),
            batch_size=1,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=float(guidance_scale),
        ).images

        # memory_management.load_models_to_gpu([vae])
        latents = (
            latents.to(dtype=self.vae.dtype, device=self.vae.device)
            / self.vae.config.scaling_factor
        )
        pixels = self.vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)

        image = Image.fromarray(pixels[0])

        out_path = "/tmp/out.png"
        image.save(out_path)

        return ModelOutput(code=code_content, image=Path(out_path))


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results
