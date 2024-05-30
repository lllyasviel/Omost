# Omost

Omost is a project to convert LLM's coding capability to image generation (or more accurately, image composing) capability. 

The name `Omost` (pronunciation: almost) has two meanings: 1) everytime after you use Omost, your image is almost there; 2) the `O` mean "omni" (multi-model) and `most` means we want to get most of it.

Omost provides LLMs models that will write codes to compose image visual contents with Omost's virtual `Canvas` agent. This `Canvas` can be rendered by specific implementations of image generators to actually generate images.

Currently, we provide 3 pretrained LLM models based on Llama3 and Phi3 (see also the model notes at the end of this page).

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

### User

generate an image of the fierce battle of warriors and dinosaurs

### Assistance (model response)

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

### Assistance (model response)

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
