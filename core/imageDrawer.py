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
    thickness: int = 4,
    display_str_list=()
):
    """Adds a bounding box to an image."""
    # TODO make display_str_list simpler -> why is there a empty tuple there?

    # draw is an editable image where we can draw stuffs, any changes in draw also gets reflected in the image
    draw = ImageDraw.Draw(image)
    # im_width, im_height = image.size

    # xmin, xmax, ymin, ymax -> values between 0 and 1 according to whole image resolution
    (left, right, top, bottom) = tuple(
        map(lambda val: round(float(val)), [xmin, xmax, ymin, ymax]))

    # draw rectangle in the bounding box
    draw.line(
        # top-left to bottom-left  to bottom-right to top-right to top-left -> rectangle
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color
    )

    # display_str_list is the list of possible objects detected by the object detection model

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(dis_str)[1]
                           for dis_str in display_str_list]
    # Each dis_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)

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

        # repeat in case of multiple detection model
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_labels, scores, max_boxes=20, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""

    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

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
                image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[
                    display_str]
            )
            np.copyto(image, np.array(image_pil))
    return image
