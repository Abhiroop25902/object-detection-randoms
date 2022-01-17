# for image IO read and decode
import tensorflow as tf

# for images save
import matplotlib.pyplot as plt

#for making image file in /tmp
import tempfile

# for downloading images from a given link
from six.moves.urllib.request import urlopen
from six import BytesIO

# for converting input photo to array form 
from PIL import Image

#for resizing images
from PIL import ImageOps

def load_img(path: str):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def save_img(image, outputFileName='./output.jpg'):
    plt.imsave(outputFileName, image)
    print(f"Images is saved as {outputFileName}")


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
