import os

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = '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'  # Remember to invalid this token when public repo

import lib_omost.memory_management as memory_management

import torch
import gradio as gr

from threading import Thread

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel
Phi3PreTrainedModel._supports_sdpa = True

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

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

memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

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


css = '''code {white-space: pre-wrap !important;}'''

chatbot = gr.Chatbot(label='Omost', scale=1, bubble_full_width=True)

with gr.Blocks(fill_height=True, css=css) as demo:
    gr.ChatInterface(
        fn=chat_fn,
        chatbot=chatbot,
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
    demo.launch(server_name='0.0.0.0')
