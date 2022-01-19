import tensorflow as tf
# For running inference on the TF-Hub module.
import tensorflow_hub as hub

# for calulating time per processing
import time

# for setting enviornment variables
import os

MODEL_HANDLE = "https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1"

# NOTE: comment this line to make the CUDA GPU
# (if proper dependencies are installed) visible to tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Detector:
    """Class wrapper for running object detector model inferences"""

    def __init__(self):
        """loads the detector model"""
        self.detector = hub.load(MODEL_HANDLE)

    def run_detector(self, img):
        """
        Run detector algorithm and returns a dictionary with keys
        "detection_boxes", "detection_classes" and "detection_scores"
        """

        # convert the image
        converted_img = tf.image.convert_image_dtype(
            img, tf.uint8
        )[tf.newaxis, ...]

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

        return result


if __name__ == "__main__":
    # Print Tensorflow version
    print(tf.__version__)

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
