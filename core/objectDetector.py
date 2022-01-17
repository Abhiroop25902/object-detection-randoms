import tensorflow as tf
# For running inference on the TF-Hub module.
import tensorflow_hub as hub

# for calulating time per processing
import time

from .imageDrawer import draw_boxes
from .imageUtils import save_img

MODEL_HANDLE = "https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1"


class Detector:
    """Class wrapper for running object detector model inferences"""
    def __init__(self):
        """loads the detector model"""
        self.detector = hub.load(MODEL_HANDLE)

    def run_detector(self, img):
        """run detector algorithm and save the image"""
        #TODO Return Image
        #TODO decouple prediction and bounding box drawing

        #convert the image 
        converted_img = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]

        start_time = time.time()
        boxes, scores, classes, num_detections = self.detector(converted_img)
        end_time = time.time()

        result = {
            'detection_boxes': boxes[0],
            'detection_scores': scores[0],
            'detection_classes': list(map(int, classes[0]))
        }

        print(f"Found {int(num_detections)} objects.")
        print(f"Inference time: {end_time-start_time} s")

        image_with_boxes = draw_boxes(
            img.numpy(),
            result["detection_boxes"],
            result["detection_classes"],
            result["detection_scores"]
        )

        save_img(image_with_boxes)
