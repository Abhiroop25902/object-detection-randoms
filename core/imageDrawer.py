# for drwaing in existing image array
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageColor

# for numerical calculation
import numpy as np

from .cocoLabelName import CocoLabelName


def draw_bounding_box_on_image(
    image: Image,
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    color: str,
    font: ImageFont,
    display_str: str = "",
    thickness: int = 4
):
    """
    Adds a bounding box to an image. caling this function
    changes the passed image itself
    """

    # draw is an editable image where we can draw stuffs, any changes in
    # draw also gets reflected in the image
    draw = ImageDraw.Draw(image)

    # xmin, xmax, ymin, ymax are float values depicting pixels of the image,
    # so rounding them to int
    (left, right, top, bottom) = tuple(
        map(lambda val: round(float(val)), [xmin, xmax, ymin, ymax])
    )

    # draw rectangle in the bounding box
    draw.line(
        # top-left to bottom-left  to bottom-right to
        # top-right to top-left -> rectangle
        [
            (left, top),
            (left, bottom),
            (right, bottom),
            (right, top),
            (left, top)
        ],
        width=thickness,
        fill=color
    )

    # If the total height of the display strings added to the top of
    # the bounding box exceeds the top of the image, stack the strings
    # below the bounding box instead of above.
    strLeft, strTop, strRight, strBottom = font.getbbox(display_str)
    text_width = strRight - strLeft
    text_height = strBottom - strTop
    # display_str has a top and bottom margin of 0.05x.
    margin = np.ceil(0.05 * text_height)
    total_display_str_height = text_height + 2*margin

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    # generate background for the text
    draw.rectangle(
        [
            (left, text_bottom - text_height - 2 * margin),  # top left
            (left + text_width, text_bottom)  # bottom right
        ],
        fill=color
    )

    # draw text
    draw.text(
        (left + margin, text_bottom - text_height - margin),  # bottom left
        display_str,
        fill="black",
        font=font
    )


def draw_boxes(
    image,
    boxes,
    class_labels,
    scores,
    max_boxes=20,
    min_score=0.1
):
    """
    Overlay labeled boxes on an image with formatted scores and label names.
    """

    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15
        )
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()
    # for converting cocoLabel to object names
    cocoLabelName = CocoLabelName()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])

            try:
                object_name = cocoLabelName.getName(class_labels[i])
            except KeyError:
                display_str = "unknown"
                print(f"unknown label found: {class_labels[i]}")

            display_str = f"{object_name}: {int(100 * scores[i])}%"
            color = colors[hash(object_name) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil, ymin, xmin, ymax, xmax, color, font, display_str
            )
            np.copyto(image, np.array(image_pil))
    return image
