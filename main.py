# For running inference on the TF-Hub module.
from email.mime import image
from shutil import register_unpack_format
import tensorflow as tf
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile

from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

# # Print Tensorflow version
# print(tf.__version__)

# # Check available GPU devices.
# print("The following GPU devices are available: %s" % tf.test.gpu_device_name())


def download_and_resize_image(url: str, new_width: int = 256, new_height: int = 256) -> str:
    """downloads and saves the image on a temporary file, return the path of that file"""
    # make a temp file to download and store the images
    _, filename = tempfile.mkstemp(suffix=".jpg")

    # download the image
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    # image is now 0 and 1 data
    # convert image to matrix form
    pil_image = Image.open(image_data)
    # resize the image
    pil_image = ImageOps.fit(
        pil_image, (new_width, new_height),
        Image.ANTIALIAS
    )

    # if image is in some different format, convert to rgb
    pil_image_rgb = pil_image.convert("RGB")

    # save file to the temporary file made
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)

    return filename


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
    # draw is an editable image where we can draw stuffs, any changes in draw also gets reflected in the image
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    # xmin, xmax, ymin, ymax -> values between 0 and 1 according to whole image resolution
    (left, right, top, bottom) = (
        xmin * im_width,
        xmax * im_width,
        ymin * im_height,
        ymax * im_height
    )

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


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(
                class_names[i].decode("ascii"),
                int(100 * scores[i])
            )
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str]
            )
            np.copyto(image, np.array(image_pil))
    return image


def load_img(path:str):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def save_img(image, outputFileName='./output.jpg'):
    plt.imsave(outputFileName, image)
    print(f"Images is saved as {outputFileName}")


def run_detector(detector, path):
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {
        key: value.numpy() for key, value in result.items()
    }

    print(result.keys())

    print("Found %d objects." % len(result["detection_scores"]))
    print(f"Inference time: {end_time-start_time} s")

    image_with_boxes = draw_boxes(
        img.numpy(),
        result["detection_boxes"],
        result["detection_class_entities"],
        result["detection_scores"]
    )

    save_img(image_with_boxes)


# By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Naxos_Taverna.jpg/800px-Naxos_Taverna.jpg"
downloaded_image_path = download_and_resize_image(image_url, 1280, 856)

# Download and load the model
module_handle = "1https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/"
detector = hub.load(module_handle).signatures['default']

if __name__ == '__main__':
    run_detector(detector, downloaded_image_path)
