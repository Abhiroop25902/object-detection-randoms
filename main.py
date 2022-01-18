from core.imageUtils import download_and_resize_image, load_img
from core.objectDetector import Detector

# By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Naxos_Taverna.jpg/800px-Naxos_Taverna.jpg"

if __name__ == '__main__':
    downloaded_image_path = download_and_resize_image(IMAGE_URL, 1280, 856)
    img = load_img(downloaded_image_path)

    detector = Detector()
    detector.run_detector(img)