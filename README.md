# Omost

Omost is a project to convert LLM's coding capability to image generation (or more accurately, image composing) capability. 

The name `Omost` (pronunciation: almost) has two meanings: 1) everytime after you use Omost, your image is almost there; 2) the `O` mean "omni" (multi-model) and `most` means we want to get most of it.

Omost provides LLMs models that will write codes to compose image visual contents with Omost's virtual `Canvas` agent. This `Canvas` can be rendered by specific implementations of image generators to actually generate images.

Currently, we provide 3 pretrained LLM models based on Llama3 and Phi3 (see also the model notes at the end of this page).

# Get Started

Below script will run the text-conditioned relighting model:

    git clone https://github.com/lllyasviel/Omost.git
    cd Omost
    conda create -n omost python=3.10
    conda activate omost
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python gradio_app.py

# Screenshots

# Symbols

# Baseline Implementation

# Examples

# Model Notes

Currently, we provide 3 models (you can get them by adding the prefix `https://huggingface.co/lllyasviel/` to the below names):

    omost-llama-3-8b
    omost-dolphin-2.9-llama3-8b
    omost-phi-3-mini-128k

And their quant versions:

    omost-llama-3-8b-4bits
    omost-dolphin-2.9-llama3-8b-4bits
    omost-phi-3-mini-128k-4bits

Some notes:

1. The recommended quant for `llama-3-8b` is 4bits, and for `phi-3-mini-128k` (3.8B) is 8 bits. They all fit in 8GB VRAM without offloads. However, quant `phi-3-mini-128k` into 4 bits is not recommended since I noticed some obvious performance degradation. The 4bit inference of `phi-3-mini-128k` should be viewed as a last method in extreme cases when you really do not have more capable GPUs.
2. My user study shows that `omost-llama-3-8b-4bits` > `omost-dolphin-2.9-llama3-8b-4bits` > `omost-phi-3-mini-128k-4bits`. So in most cases one should just use `omost-llama-3-8b-4bits`.
3. The `omost-llama-3-8b-4bits` and `omost-phi-3-mini-128k-4bits` are trained with filtered safe data without NSFW or inappropriate contents. The `omost-dolphin-2.9-llama3-8b-4bits` is trained with all data without filtering.
4. Note that the filtering in 3 is 
