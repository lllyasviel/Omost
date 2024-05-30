# Omost

Omost is a project to convert LLM's coding capability to image generation (or more accurately, image composing) capability. 

The name `Omost` (pronunciation: almost) has two meanings: 1) everytime after you use Omost, your image is almost there; 2) the `O` mean "omni" (multi-modal) and `most` means we want to get most of it.

Omost provides LLMs models that will write codes to compose image visual contents with Omost's virtual `Canvas` agent. This `Canvas` can be rendered by specific implementations of image generators to actually generate images.

Currently, we provide 3 pretrained LLM models based on Llama3 and Phi3 (see also the model notes at the end of this page).

All models are trained with mixed data of (1) ground-truth annotations of several datasets including Open-Images, (2) extracted data by automatically annotating images, (3) reinforcement from DPO (Direct Preference Optimization, "whether the codes can be compiled by python 3.10 or not" as Direct Preference), and (4) a small amount of tuning data from OpenAI GPT4o's multi-modal capability.

# Get Started

You can just use the [official HuggingFace space](https://huggingface.co/spaces/lllyasviel/Omost).

Or, you can use the below deployment (requires 8GB Nvidia VRAM):

    git clone https://github.com/lllyasviel/Omost.git
    cd Omost
    conda create -n omost python=3.10
    conda activate omost
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python gradio_app.py

(Note that quant LLM requires `bitsandbytes` - some 9XX or 10XX or 20XX GPUs may have trouble in running it. If that happens, just use our official huggingface space.)

# Screenshots

*(Below is a raw transcript from default model)*

*(Random seed is 123456)*

*(If the GitHub's code block rendering is causing troubles to read long lines, click [here](https://raw.githubusercontent.com/lllyasviel/Omost/main/README.md) to read raw texts.)*

<details>
<summary>Screenshots</summary>
<img src="https://github.com/lllyasviel/Omost/assets/19834515/0b3479c8-a157-4c4e-b865-d32355da6bc6"  alt=""/>
</details>

### User

generate an image of the fierce battle of warriors and dinosaurs

### Assistant (model response)

```python
# Initialize the canvas
canvas = Canvas()

# Set a global description for the canvas
canvas.set_global_description(
    description='An intense battle scene featuring warriors and dinosaurs.',
    detailed_descriptions=[
        'In this dramatic scene, warriors clad in ancient armor engage in a fierce battle with massive dinosaurs.',
        'The warriors, wielding swords and shields, are positioned strategically to counter the dinosaurs, which are a mix of fierce predators and gentle herbivores.',
        'The dinosaurs are depicted in various stages of attack and defense, with some roaring and others charging.',
        'The background features a lush, prehistoric landscape with dense forests, towering trees, and a misty atmosphere.',
        'The sky is overcast with dark, ominous clouds, adding to the intensity of the battle.',
        'The ground is uneven, with patches of mud and fallen trees, creating obstacles for both the warriors and dinosaurs.',
        'The overall mood is one of chaos and adrenaline, capturing the raw power and bravery of the warriors and the primal fury of the dinosaurs.',
    ],
    tags='battle, warriors, dinosaurs, ancient armor, swords, shields, fierce predators, gentle herbivores, lush landscape, prehistoric, dense forests, towering trees, misty atmosphere, overcast sky, dark clouds, uneven ground, mud patches, fallen trees, chaos, adrenaline, raw power, bravery, primal fury',
    HTML_web_color_name='darkslategray',
)

# Add a group of warriors fighting a dinosaur.
canvas.add_local_description(
    location='in the center',
    offset='no offset',
    area='a large square area',
    distance_to_viewer=5.0,
    description='A group of warriors fighting a dinosaur.',
    detailed_descriptions=[
        'In the heart of the scene, a group of warriors is engaged in a fierce battle with a massive dinosaur.',
        'The warriors are clad in ancient, ornate armor, with intricate designs and battle scars.',
        'They wield swords and shields, their movements a blur of aggression and skill.',
        'The dinosaur, a towering T-Rex, is in mid-roar, its massive jaws open wide.',
        'Its body is muscular and covered in rough, scaly skin.',
        'The ground around them is churned up, with mud flying and debris scattered.',
        'The intensity of this central fight is palpable, capturing the raw energy and primal nature of the combatants.',
    ],
    tags='warriors, dinosaur, battle, ancient armor, swords, shields, T-Rex, mid-roar, muscular, scaly skin, mud, debris, intensity, raw energy, primal nature',
    atmosphere='The atmosphere is one of intense combat and raw power.',
    style='The style is dynamic and action-packed, with a focus on movement and detail.',
    quality_meta='High-quality depiction with detailed textures and dynamic lighting.',
    HTML_web_color_name='darkolivegreen',
)

# Add a warrior defending against a stegosaurus.
canvas.add_local_description(
    location='on the left',
    offset='slightly to the lower',
    area='a medium-sized vertical area',
    distance_to_viewer=6.5,
    description='A warrior defending against a Stegosaurus.',
    detailed_descriptions=[
        'On the left side of the scene, a warrior is seen defending against a Stegosaurus.',
        'The warrior, clad in leather armor, is positioned behind a wooden shield, using it to deflect the dinosaur’s attacks.',
        'The Stegosaurus, with its distinctive row of plates running down its back, is in the midst of a defensive stance.',
        'Its long tail swings back, ready to strike.',
        'The surrounding area is littered with fallen branches and scattered rocks, adding to the chaotic environment.',
        'The warrior’s expression is one of determination and focus, embodying the spirit of the battle.',
    ],
    tags='warrior, Stegosaurus, defense, leather armor, wooden shield, deflect attacks, row of plates, defensive stance, long tail, fallen branches, scattered rocks, chaotic, determination, focus',
    atmosphere='The atmosphere is tense and focused, with a sense of impending action.',
    style='The style is detailed and realistic, with a focus on the defensive actions.',
    quality_meta='High-quality depiction with realistic textures and dynamic lighting.',
    HTML_web_color_name='saddlebrown',
)

# Add a triceratops charging at warriors.
canvas.add_local_description(
    location='on the right',
    offset='slightly to the upper',
    area='a medium-sized vertical area',
    distance_to_viewer=7.0,
    description='A Triceratops charging at warriors.',
    detailed_descriptions=[
        'On the right side of the scene, a Triceratops is charging at a group of warriors.',
        'The Triceratops, with its distinctive frill and horns, is depicted in a full-on charge, its body low to the ground and ready to attack.',
        'The warriors, clad in chainmail and leather, are scattered in a defensive formation, their weapons at the ready.',
        'The ground beneath the Triceratops is churned up, with dirt flying and the warriors’ formation broken.',
        'The surrounding area is a mix of dense forest and open ground, with fallen trees and underbrush adding to the chaos.',
        'The charging Triceratops creates a sense of urgency and danger, highlighting the primal nature of the dinosaurs.',
    ],
    tags='Triceratops, charging, warriors, distinctive frill, horns, full-on charge, low to the ground, attack, chainmail, leather, defensive formation, weapons, churned ground, dirt flying, broken formation, dense forest, open ground, fallen trees, underbrush, chaos, urgency, danger, primal nature',
    atmosphere='The atmosphere is one of urgency and danger, with a sense of primal chaos.',
    style='The style is dynamic and action-packed, with a focus on movement and chaos.',
    quality_meta='High-quality depiction with detailed textures and dynamic lighting.',
    HTML_web_color_name='darkgreen',
)

# Add a pterodactyl soaring above the battle.
canvas.add_local_description(
    location='on the top',
    offset='slightly to the right',
    area='a small horizontal area',
    distance_to_viewer=8.0,
    description='A Pterodactyl soaring above the battle.',
    detailed_descriptions=[
        'Above the battle, a Pterodactyl soars through the dark, overcast sky.',
        'The Pterodactyl, with its large wings and sharp claws, is depicted in mid-flight, its eyes scanning the battlefield below.',
        'The sky around it is filled with dark clouds, and occasional flashes of lightning illuminate the scene.',
        'The presence of the Pterodactyl adds a sense of height and scale to the battle, emphasizing the vast and unpredictable nature of the prehistoric world.',
        'The overall mood is one of ominous grandeur, as the Pterodactyl’s silent vigil adds to the tension below.',
    ],
    tags='Pterodactyl, soaring, dark sky, overcast, large wings, sharp claws, mid-flight, eyes scanning, dark clouds, lightning, height, scale, battle, vast, unpredictable, prehistoric, ominous, grandeur, silent vigil, tension',
    atmosphere='The atmosphere is one of ominous grandeur, with a sense of height and tension.',
    style='The style is majestic and atmospheric, with a focus on scale and tension.',
    quality_meta='High-quality depiction with detailed textures and atmospheric lighting.',
    HTML_web_color_name='slategray',
)

# Add a fallen warrior and dinosaur.
canvas.add_local_description(
    location='on the bottom-left',
    offset='slightly to the upper-right',
    area='a small square area',
    distance_to_viewer=4.5,
    description='A fallen warrior and dinosaur.',
    detailed_descriptions=[
        'In the bottom-left corner, a fallen warrior and dinosaur are depicted in a moment of defeat.',
        'The warrior, clad in tattered armor, lies on the ground, his sword slipping from his grasp.',
        'The dinosaur, a gentle Apatosaur, is seen standing over the warrior, its long neck and tail forming a protective stance.',
        'The surrounding area is littered with fallen debris, including branches and rocks, adding to the sense of chaos and loss.',
        'The fallen warrior’s expression is one of resignation and sorrow, capturing the tragic end of the battle.',
        'The overall mood is one of sorrow and loss, highlighting the cost of the fierce battle.',
    ],
    tags='fallen warrior, dinosaur, defeat, tattered armor, sword, slipping grasp, Apatosaur, standing over, protective stance, fallen debris, branches, rocks, chaos, loss, resignation, sorrow, tragic end, battle, cost, fierce',
    atmosphere='The atmosphere is one of sorrow and loss, with a sense of tragic defeat.',
    style='The style is somber and detailed, with a focus on the tragic end.',
    quality_meta='High-quality depiction with detailed textures and somber lighting.',
    HTML_web_color_name='dimgray',
)
```

### User

add some meteorites to the picture

### Assistant (model response)

```python
# Initialize the canvas
canvas = Canvas()

# Set a global description for the canvas
canvas.set_global_description(
    description='An intense battle scene featuring warriors, dinosaurs, and meteorites.',
    detailed_descriptions=[
        'In this dramatic scene, warriors clad in ancient armor engage in a fierce battle with massive dinosaurs, while meteorites fall from the sky.',
        'The warriors, wielding swords and shields, are positioned strategically to counter the dinosaurs, which are a mix of fierce predators and gentle herbivores.',
        'The dinosaurs are depicted in various stages of attack and defense, with some roaring and others charging.',
        'The background features a lush, prehistoric landscape with dense forests, towering trees, and a misty atmosphere.',
        'The sky is overcast with dark, ominous clouds, adding to the intensity of the battle. Meteorites are seen crashing into the ground, creating explosions and sending debris flying.',
        'The ground is uneven, with patches of mud and fallen trees, creating obstacles for both the warriors and dinosaurs.',
        'The overall mood is one of chaos and adrenaline, capturing the raw power and bravery of the warriors, the primal fury of the dinosaurs, and the destructive force of the meteorites.',
    ],
    tags='battle, warriors, dinosaurs, ancient armor, swords, shields, fierce predators, gentle herbivores, lush landscape, prehistoric, dense forests, towering trees, misty atmosphere, overcast sky, dark clouds, uneven ground, mud patches, fallen trees, chaos, adrenaline, raw power, bravery, primal fury, meteorites, explosions, debris, destruction',
    HTML_web_color_name='darkslategray',
)

# Add a group of warriors fighting a dinosaur.
canvas.add_local_description(
    location='in the center',
    offset='no offset',
    area='a large square area',
    distance_to_viewer=5.0,
    description='A group of warriors fighting a dinosaur.',
    detailed_descriptions=[
        'In the heart of the scene, a group of warriors is engaged in a fierce battle with a massive dinosaur.',
        'The warriors are clad in ancient, ornate armor, with intricate designs and battle scars.',
        'They wield swords and shields, their movements a blur of aggression and skill.',
        'The dinosaur, a towering T-Rex, is in mid-roar, its massive jaws open wide.',
        'Its body is muscular and covered in rough, scaly skin.',
        'The ground around them is churned up, with mud flying and debris scattered.',
        'The intensity of this central fight is palpable, capturing the raw energy and primal nature of the combatants.',
    ],
    tags='warriors, dinosaur, battle, ancient armor, swords, shields, T-Rex, mid-roar, muscular, scaly skin, mud, debris, intensity, raw energy, primal nature',
    atmosphere='The atmosphere is one of intense combat and raw power.',
    style='The style is dynamic and action-packed, with a focus on movement and detail.',
    quality_meta='High-quality depiction with detailed textures and dynamic lighting.',
    HTML_web_color_name='darkolivegreen',
)

# Add a warrior defending against a stegosaurus.
canvas.add_local_description(
    location='on the left',
    offset='slightly to the lower',
    area='a medium-sized vertical area',
    distance_to_viewer=6.5,
    description='A warrior defending against a Stegosaurus.',
    detailed_descriptions=[
        'On the left side of the scene, a warrior is seen defending against a Stegosaurus.',
        'The warrior, clad in leather armor, is positioned behind a wooden shield, using it to deflect the dinosaur’s attacks.',
        'The Stegosaurus, with its distinctive row of plates running down its back, is in the midst of a defensive stance.',
        'Its long tail swings back, ready to strike.',
        'The surrounding area is littered with fallen branches and scattered rocks, adding to the chaotic environment.',
        'The warrior’s expression is one of determination and focus, embodying the spirit of the battle.',
    ],
    tags='warrior, Stegosaurus, defense, leather armor, wooden shield, deflect attacks, row of plates, defensive stance, long tail, fallen branches, scattered rocks, chaotic, determination, focus',
    atmosphere='The atmosphere is tense and focused, with a sense of impending action.',
    style='The style is detailed and realistic, with a focus on the defensive actions.',
    quality_meta='High-quality depiction with realistic textures and dynamic lighting.',
    HTML_web_color_name='saddlebrown',
)

# Add a triceratops charging at warriors.
canvas.add_local_description(
    location='on the right',
    offset='slightly to the upper',
    area='a medium-sized vertical area',
    distance_to_viewer=7.0,
    description='A Triceratops charging at warriors.',
    detailed_descriptions=[
        'On the right side of the scene, a Triceratops is charging at a group of warriors.',
        'The Triceratops, with its distinctive frill and horns, is depicted in a full-on charge, its body low to the ground and ready to attack.',
        'The warriors, clad in chainmail and leather, are scattered in a defensive formation, their weapons at the ready.',
        'The ground beneath the Triceratops is churned up, with dirt flying and the warriors’ formation broken.',
        'The surrounding area is a mix of dense forest and open ground, with fallen trees and underbrush adding to the chaos.',
        'The charging Triceratops creates a sense of urgency and danger, highlighting the primal nature of the dinosaurs.',
    ],
    tags='Triceratops, charging, warriors, distinctive frill, horns, full-on charge, low to the ground, attack, chainmail, leather, defensive formation, weapons, churned ground, dirt flying, broken formation, dense forest, open ground, fallen trees, underbrush, chaos, urgency, danger, primal nature',
    atmosphere='The atmosphere is one of urgency and danger, with a sense of primal chaos.',
    style='The style is dynamic and action-packed, with a focus on movement and chaos.',
    quality_meta='High-quality depiction with detailed textures and dynamic lighting.',
    HTML_web_color_name='darkgreen',
)

# Add a pterodactyl soaring above the battle.
canvas.add_local_description(
    location='on the top',
    offset='slightly to the right',
    area='a small horizontal area',
    distance_to_viewer=8.0,
    description='A Pterodactyl soaring above the battle.',
    detailed_descriptions=[
        'Above the battle, a Pterodactyl soars through the dark, overcast sky.',
        'The Pterodactyl, with its large wings and sharp claws, is depicted in mid-flight, its eyes scanning the battlefield below.',
        'The sky around it is filled with dark clouds, and occasional flashes of lightning illuminate the scene.',
        'The presence of the Pterodactyl adds a sense of height and scale to the battle, emphasizing the vast and unpredictable nature of the prehistoric world.',
        'The overall mood is one of ominous grandeur, as the Pterodactyl’s silent vigil adds to the tension below.',
    ],
    tags='Pterodactyl, soaring, dark sky, overcast, large wings, sharp claws, mid-flight, eyes scanning, dark clouds, lightning, height, scale, battle, vast, unpredictable, prehistoric, ominous, grandeur, silent vigil, tension',
    atmosphere='The atmosphere is one of ominous grandeur, with a sense of height and tension.',
    style='The style is majestic and atmospheric, with a focus on scale and tension.',
    quality_meta='High-quality depiction with detailed textures and atmospheric lighting.',
    HTML_web_color_name='slategray',
)

# Add a fallen warrior and dinosaur.
canvas.add_local_description(
    location='on the bottom-left',
    offset='slightly to the upper-right',
    area='a small square area',
    distance_to_viewer=4.5,
    description='A fallen warrior and dinosaur.',
    detailed_descriptions=[
        'In the bottom-left corner, a fallen warrior and dinosaur are depicted in a moment of defeat.',
        'The warrior, clad in tattered armor, lies on the ground, his sword slipping from his grasp.',
        'The dinosaur, a gentle Apatosaur, is seen standing over the warrior, its long neck and tail forming a protective stance.',
        'The surrounding area is littered with fallen debris, including branches and rocks, adding to the sense of chaos and loss.',
        'The fallen warrior’s expression is one of resignation and sorrow, capturing the tragic end of the battle.',
        'The overall mood is one of sorrow and loss, highlighting the cost of the fierce battle.',
    ],
    tags='fallen warrior, dinosaur, defeat, tattered armor, sword, slipping grasp, Apatosaur, standing over, protective stance, fallen debris, branches, rocks, chaos, loss, resignation, sorrow, tragic end, battle, cost, fierce',
    atmosphere='The atmosphere is one of sorrow and loss, with a sense of tragic defeat.',
    style='The style is somber and detailed, with a focus on the tragic end.',
    quality_meta='High-quality depiction with detailed textures and somber lighting.',
    HTML_web_color_name='dimgray',
)

# Add meteorites crashing into the ground.
canvas.add_local_description(
    location='on the top-right',
    offset='slightly to the lower-left',
    area='a small square area',
    distance_to_viewer=9.0,
    description='Meteorites crashing into the ground.',
    detailed_descriptions=[
        'On the top-right side of the scene, meteorites are seen crashing into the ground.',
        'The meteorites, glowing with intense heat, are depicted mid-crash, sending flames and debris flying.',
        'The ground around the meteorites is scorched, with craters forming and smoke rising.',
        'The surrounding area is filled with the sounds of explosions and the sight of burning debris.',
        'The meteorites add a sense of danger and unpredictability to the already chaotic battle scene.',
    ],
    tags='meteorites, crashing, ground, intense heat, mid-crash, flames, debris, scorched, craters, smoke, explosions, burning debris, danger, unpredictability, chaotic, battle',
    atmosphere='The atmosphere is one of danger and unpredictability, with a sense of chaotic destruction.',
    style='The style is dramatic and intense, with a focus on the destructive force of the meteorites.',
    quality_meta='High-quality depiction with dynamic textures and dramatic lighting.',
    HTML_web_color_name='firebrick',
)
```

(User can click the render button to actually render the image from any of those `Canvas` instances.)

<details>
<summary>Screenshots</summary>
<img src="https://github.com/lllyasviel/Omost/assets/19834515/eaddedef-d038-4119-8496-0f481350bf37"  alt=""/>
<img src="https://github.com/lllyasviel/Omost/assets/19834515/e6cf4120-2e95-4580-be8b-36221c0f4d35"  alt=""/>
</details>

# Symbols

All Omost LLMs are trained to obey the following symbols

```python
class Canvas:
    def set_global_description(
            self, 
            description: str, 
            detailed_descriptions: list[str], 
            tags: str, 
            HTML_web_color_name: str
    ):
        pass

    def add_local_description(
            self, 
            location: str, 
            offset: str, 
            area: str, 
            distance_to_viewer: float, 
            description: str, 
            detailed_descriptions: list[str], 
            tags: str, 
            atmosphere: str, 
            style: str, 
            quality_meta: str, 
            HTML_web_color_name: str
    ):
        assert location in [
            "in the center", 
            "on the left", 
            "on the right", 
            "on the top", 
            "on the bottom", 
            "on the top-left", 
            "on the top-right", 
            "on the bottom-left", 
            "on the bottom-right"
        ]
        assert offset in [
            "no offset", 
            "slightly to the left", 
            "slightly to the right", 
            "slightly to the upper", 
            "slightly to the lower", 
            "slightly to the upper-left", 
            "slightly to the upper-right", 
            "slightly to the lower-left", 
            "slightly to the lower-right"
        ]
        assert area in [
            "a small square area", 
            "a small vertical area", 
            "a small horizontal area", 
            "a medium-sized square area", 
            "a medium-sized vertical area", 
            "a medium-sized horizontal area", 
            "a large square area", 
            "a large vertical area", 
            "a large horizontal area"
        ]
        assert distance_to_viewer > 0
        pass
```

During training, the above symbols are associated with specific concepts and use cases related to image generation.

The design is to make those codes easy to learn for LLMs, but also easy to handle for diffusion models.

Lets breakdown each part:

## Function: Canvas.set_global_description and Canvas.add_local_description

They set descriptions to images. The meanings of the parameters are same for them, with `add_local_description` having more fields than `set_global_description`.

The `set_global_description` annotate entire image, while `add_local_description` annotates a part of image.

## Parameter: description and detailed_descriptions

Let us introduce a concept called "sub-prompt". If a prompt (1) is less than 75 tokens, and (2) is self-supported to describe a thing without relying on other prompts, we call it a "sub-prompt".

The `description` is a sub-prompt, and the `detailed_descriptions` is a list of sub-prompts.

Note that each sub-prompt is strictly less than 75 tokens (and typically less than 40 tokens), you can safely encode them with any clip without worrying the truncation position affecting the semantics.

The design of sub-prompt also allows more satisfying text encoding based on greedy merge. For example, if you have 

    sub-prompt A: 25 tokens
    sub-prompt B: 35 tokens
    sub-prompt C: 5 tokens
    sub-prompt D: 60 tokens
    sub-prompt E: 15 tokens
    sub-prompt F: 25 tokens

and since every sub-prompt is promised to be self-supported to describe a thing independently, we can use greedy method to merge them to bags like

    bag 1 {A, B, C} : 65 tokens
    bag 2 {D} : 60 tokens
    bag 1 {E, F} : 40 tokens

where each bag is less than 75 tokens and can be encoded by any clip in one pass (and then concat them). 

Encoding texts in this way will make sure that text-encoder will never make semantic truncation mistakes. 

One may ask - if all sub-prompts are less than 75 tokens with independent semantics, why not just encode them without merge and then concat? This is mainly because we want the text embedding to be more coherent. For example, lets say sub-prompt A is "a man" while sub-prompt B is "handsome, professional", then merging them before encoding will give you a more mixed text embedding concept with coherent features of a handsome professional man. 

All Omost LLMs are trained to give strictly well-defined sub-prompts. You can make use of these definitions to design lossless text encoding methods.

### Parameter: location, offset, area

The three parameters defines a bounding box. Note that they must obey

```python
assert location in [
    "in the center", 
    "on the left", 
    "on the right", 
    "on the top", 
    "on the bottom", 
    "on the top-left", 
    "on the top-right", 
    "on the bottom-left", 
    "on the bottom-right"
]
assert offset in [
    "no offset", 
    "slightly to the left", 
    "slightly to the right", 
    "slightly to the upper", 
    "slightly to the lower", 
    "slightly to the upper-left", 
    "slightly to the upper-right", 
    "slightly to the lower-left", 
    "slightly to the lower-right"
]
assert area in [
    "a small square area", 
    "a small vertical area", 
    "a small horizontal area", 
    "a medium-sized square area", 
    "a medium-sized vertical area", 
    "a medium-sized horizontal area", 
    "a large square area", 
    "a large vertical area", 
    "a large horizontal area"
]
```

First we divide a canvas into 3*3=9 locations:

![image](https://github.com/lllyasviel/Omost/assets/19834515/5d39cf93-c229-4c83-ae82-3eeeae2fabea)

Then we further divide each location to 3\*3 offsets, resulting in 9\*9=81 positions:

![image](https://github.com/lllyasviel/Omost/assets/19834515/b744d787-11f3-4aeb-9d3a-aeba7a41b433)

Using these positions as centers, we further define 9 types of bounding boxes:

![image](https://github.com/lllyasviel/Omost/assets/19834515/0e484b73-680f-486b-8b61-4373c9eec9a0)

We can see that this method allows 9\*9\*9=729 different bounding boxes, covering almost all common possible locations of an object in the image.

One mat argue that why this is necessary - why not just let the LLMs to learn pixel index or x, y coordinates - and should that be much more accurate? Below is several of my notes:

1. I have tried several representations, including pixel index like {x=32, y=16, w=58, h=99}, or margin pixels like {left=32, right=15, top=27, bottom=33}, or percentage pixel index like {x=0.124, y=0.65, w=0.335, h=0.251}, or percentage margin like {left=0.251, right=0.154, top=0.254, bottom=0.441}. The result is that opensource LLMs are really not very good at learning these representations even for Llama3 (perhaps GPT4o can learn it). Sometimes it works sometimes it gives completely random numbers.
2. But the natural language like "on the right", "slightly to the top-right", "a small vertical area" etc, works very well. The model converges very fast and the learning is stable. It aligns to the pretrained knowledge of LLMs very well.
3. I have also tried using add some special tokens to represent spatial locations and also train the embedding layers. But that model is very difficult to train and debug. Also, this token-based method needs many hyperparameter tuning everytime we change the LLM - for example when changing from Llama3 to Phi, if we use this method, we need to design training parameters again.
4. The number 9\*9\*9=729 is really not a small number from the perspective of bounding box proposals. This can also be called ROI (region of interest) and some old semantic segmentation tech uses (RPN) Region Proposal Network to produce a similar number (<1000) of regions.
5. Most region-guided diffusion methods are coarse-level methods (like multi-diffusion and attention couple and gligen), and they do not need pixel-perfect regions.
6. These are very personal results from me - if you are working on some similar multi-modal LLM research, using pixel indices is completely okay, worth trying, and probably other training methods can also achieve robust system.

# A Baseline Renderer

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

1. The recommended quant for `omost-llama-3-8b` is 4bits, and for `omost-phi-3-mini-128k` (3.8B) is 8 bits. They all fit in 8GB VRAM without offloads. The performance degradation caused by quant is very minimal and I personally never observed any evidences of degradation. However, quant `omost-phi-3-mini-128k` into 4 bits is not recommended since I noticed some obvious performance degradation. The 4bit inference of `omost-phi-3-mini-128k` should be viewed as a last method in extreme cases when you really do not have more capable GPUs.
2. My user study shows that `omost-llama-3-8b-4bits` > `omost-dolphin-2.9-llama3-8b-4bits` > `omost-phi-3-mini-128k-4bits`. So in most cases one should just use `omost-llama-3-8b-4bits`.
3. The `omost-llama-3-8b-4bits` and `omost-phi-3-mini-128k-4bits` are trained with filtered safe data without NSFW or inappropriate contents. See (4) if you need a different option.
4. The `omost-dolphin-2.9-llama3-8b-4bits` is trained with all data WITHOUT any filtering. You must apply your own safety alignment methods if you expose any service of `omost-dolphin-2.9-llama3-8b-4bits` to public.
5. Note that the filtering in (3) is not because of any policy - the reason is that I noticed slight instability in training gradients in those models since they are pretrained with instruct following regulated by safety alignment, causing the performance to degrade a bit. But the instruct following of `omost-dolphin-2.9-llama3-8b-4bits` is pretrained with community efforts and do not have this problem.
6. The 128k context length of `omost-phi-3-mini-128k` cannot be trusted. The performance of it will degrade a lot after the tokens reach about 8k. One should just view it as a model with about 8k content length.
7. A model of 8k context length can do about 5 to 6 rounds of conversational editing. If you are about to run out of token lengths, use the UI to modify your message and respond again (this can be done with infinite times).
8. All models are fully trained with our H100 clusters at precision fp16 without any tricks like quant or Q-LoRA etc. The optimizer is Adam without any tricks.
9. You must also follow the licenses of Llama-3 and Phi-3.
10. You can request us to train on other LLMs if reasonable and necessary.
