# Initialize the canvas
canvas = Canvas()

# Set a global description for the canvas
canvas.set_global_description(
    description='A cat on a table in a room',
    detailed_descriptions=[
        'The image depicts a cozy room with a cat sitting gracefully on a wooden table.',
        'The table is well-crafted with a polished wooden surface, and the cat, with its fur well-groomed, appears relaxed and observant.',
        'The room is warmly lit, with soft daylight streaming in from a window on the left side.',
        'The walls are adorned with subtle, elegant wallpaper, and there are a few pieces of furniture including a comfortable-looking armchair and a small bookshelf filled with books.',
        'The floor is covered with a soft, plush carpet, adding to the overall comfort and warmth of the room.',
        'The cat, a sleek black feline, is positioned centrally on the table, its eyes fixed on something in the room.',
        'The overall atmosphere is calm and inviting, with a sense of tranquility and comfort.',
    ],
    tags='cat, table, room, wooden table, black cat, cozy room, daylight, window, wallpaper, armchair, bookshelf, books, plush carpet, comfortable, warm, tranquil, observant, relaxed, polished wood, well-crafted, fur, elegant, soft daylight, inviting, calm, comfort, tranquility',
    HTML_web_color_name='lightgoldenrodyellow',
)

# Add a cat sitting on a table
canvas.add_local_description(
    location='in the center',
    offset='no offset',
    area='a medium-sized square area',
    distance_to_viewer=1.5,
    description='A cat sitting on a table',
    detailed_descriptions=[
        'The central focus of the image is a sleek black cat, sitting gracefully on a wooden table.',
        'The cat appears well-groomed, with its fur meticulously arranged.',
        'It is positioned centrally on the table, its eyes wide open as if observing something in the room.',
        'The table itself is a piece of craftsmanship, with a polished wooden surface that reflects the soft daylight streaming in from the window.',
        "The cat's posture is relaxed, with its paws tucked under and its tail curled around its body.",
        'This central figure adds a sense of calm and tranquility to the overall atmosphere of the room.',
    ],
    tags='cat, black cat, wooden table, well-groomed, polished wood, craftsmanship, relaxed, observant, central focus, fur, posture, calm, tranquility, soft daylight, window, paws, tail, body, curled, tucked',
    atmosphere='Calm and tranquil, with a sense of observant curiosity.',
    style='Realistic and detailed, with a focus on the cat’s fur and posture.',
    quality_meta='High-quality rendering with detailed textures and lighting.',
    HTML_web_color_name='black',
)

# Add a window letting in daylight
canvas.add_local_description(
    location='on the left',
    offset='slightly to the upper',
    area='a medium-sized vertical area',
    distance_to_viewer=2.0,
    description='A window letting in daylight',
    detailed_descriptions=[
        'On the left side of the room, slightly to the upper part, there is a window allowing soft daylight to stream into the room.',
        'The window has a simple, elegant frame that complements the overall decor of the room.',
        'The daylight that enters through the window casts a warm, inviting glow over the room, highlighting the subtle textures and details within.',
        'This natural light enhances the cozy and comfortable atmosphere of the room, adding a sense of warmth and tranquility.',
    ],
    tags='window, daylight, soft light, warm glow, natural light, elegant frame, decor, room, warm, inviting, glow, textures, details, cozy, comfortable, atmosphere, warmth, tranquility, highlights, complements, upper, left',
    atmosphere='Warm and inviting, with a soft, natural glow.',
    style='Elegant and simple, with a focus on natural lighting.',
    quality_meta='High-quality rendering with realistic lighting and shadows.',
    HTML_web_color_name='lightyellow',
)

# Add an armchair with a small bookshelf
canvas.add_local_description(
    location='on the right',
    offset='slightly to the lower',
    area='a medium-sized vertical area',
    distance_to_viewer=2.5,
    description='An armchair with a small bookshelf',
    detailed_descriptions=[
        'To the right of the room, slightly lower than the central area, there is a comfortable-looking armchair accompanied by a small bookshelf.',
        'The armchair is upholstered in a soft, plush fabric, inviting one to sit and relax.',
        'The bookshelf, filled with various books, adds a touch of intellectual charm to the room.',
        'Both pieces of furniture are well-integrated into the room’s decor, contributing to the overall sense of comfort and tranquility.',
    ],
    tags='armchair, bookshelf, comfortable, plush fabric, relax, books, intellectual charm, furniture, room, decor, integrated, comfort, tranquility, soft, inviting, touch, lower, right, upholstered, various, well-integrated',
    atmosphere='Comfortable and tranquil, with a touch of intellectual charm.',
    style='Cozy and inviting, with a focus on comfort and intellectual appeal.',
    quality_meta='High-quality rendering with detailed textures and integrated elements.',
    HTML_web_color_name='beige',
)

# Add a plush carpet covering the floor
canvas.add_local_description(
    location='on the bottom',
    offset='slightly to the left',
    area='a large horizontal area',
    distance_to_viewer=3.0,
    description='A plush carpet covering the floor',
    detailed_descriptions=[
        'The floor of the room is covered with a soft, plush carpet that adds to the overall comfort and warmth of the room.',
        'The carpet is plush and thick, with a rich texture that enhances the cozy atmosphere.',
        'It is spread evenly across the floor, with no visible seams or edges.',
        'The carpet’s color is a soft, neutral beige, which complements the rest of the room’s decor.',
        'This element contributes significantly to the inviting and tranquil feel of the room, making it a perfect place to relax and unwind.',
    ],
    tags='plush carpet, floor, soft, comfortable, warm, rich texture, cozy, atmosphere, spread, evenly, neutral beige, decor, inviting, tranquil, feel, relax, unwind, thick, texture, room, complements, element, significant, perfect',
    atmosphere='Inviting and tranquil, with a soft, comfortable feel.',
    style='Rich and plush, with a focus on texture and comfort.',
    quality_meta='High-quality rendering with realistic textures and spread.',
    HTML_web_color_name='beige',
)

# Add subtle, elegant wallpaper
canvas.add_local_description(
    location='on the top',
    offset='no offset',
    area='a large horizontal area',
    distance_to_viewer=3.5,
    description='Subtle, elegant wallpaper',
    detailed_descriptions=[
        'The walls of the room are adorned with subtle, elegant wallpaper that adds a touch of sophistication to the room’s decor.',
        'The wallpaper is patterned with delicate, understated designs that do not overpower the room but instead complement its overall aesthetic.',
        'The color of the wallpaper is a soft, muted shade that blends harmoniously with the rest of the room’s elements.',
        'This subtle decoration enhances the room’s inviting and tranquil atmosphere, making it a visually pleasing space.',
    ],
    tags='wallpaper, walls, subtle, elegant, sophisticated, decor, patterned, delicate, understated, designs, complement, aesthetic, muted shade, harmonious, elements, decoration, enhances, inviting, tranquil, atmosphere, visually pleasing, space, room',
    atmosphere='Sophisticated and inviting, with a touch of elegance.',
    style='Delicate and understated, with a focus on subtle designs.',
    quality_meta='High-quality rendering with realistic patterns and colors.',
    HTML_web_color_name='lavender',
)