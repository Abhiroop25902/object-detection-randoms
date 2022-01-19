# for image write and read, resizing image
import cv2

# for making image file in /tmp
import tempfile

# for downloading images from a given link
import urllib
import numpy as np


def show_image(bgr_img, window_name='image'):
    '''
    shows the image in a new window and waits for the user to close
    (or for wait_duration ms if provided)
    '''
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    cv2.imshow("image", rgb_img)

    wait_time = 1000
    # while the 'image' named window is visible
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        # wait for wait_time
        keyCode = cv2.waitKey(wait_time)
        # if in the waiting duration, someone pressed q
        if (keyCode & 0xFF) == ord("q"):
            # destroy all windows
            cv2.destroyAllWindows()
            break


def load_img(path: str):
    bgr_img = cv2.imread(path)
    # NOTE: cv2.imread reads in BGR format, so we need to change it to RGB
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def save_img(rgb_img, outputFileName='./output.jpg'):
    # NOTE: img saved will be read as BRG format, we changed it to RGB in
    # load_img, so we need to convert RGB to BRG
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(outputFileName, bgr_img)
    print(f"Images is saved as {outputFileName}")


def download_and_resize_image(
    url: str,
    new_width: int = 256,
    new_height: int = 256
) -> str:
    """
    downloads and saves the image on a temporary file,
    return the path of that file
    """
    # make a temp file to download and store the images
    _, filename = tempfile.mkstemp(suffix=".jpg")

    with urllib.request.urlopen(url) as response:
        # read image as an numpy array
        image = np.asarray(bytearray(response.read()), dtype="uint8")

    # convert numpy image array to cv2 image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # resize the image
    image = cv2.resize(
        image,
        (new_width, new_height)
    )
    # NOTE: img is in BGR format and imwrite also reads it in BRG format

    # save file to the temporary file made
    cv2.imwrite(filename, image)
    print("Image downloaded to %s." % filename)

    return filename