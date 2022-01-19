import time
from core.imageDrawer import draw_boxes
import cv2
from core.objectDetector import Detector


def run_camera(cap, detector, inference_timings):
    # while the capture element is open (video has next frames)
    # and the display window is visible (not closed)
    while (
        cap.isOpened()
        and cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) >= 1
    ):
        # get frame from teh capture
        ret, frame = cap.read()

        # there is some error in getting the next frame so we break the loop
        if ret is False:
            break

        # change the image to rgb
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # start timer for the inference
        start_time = time.time()
        # run the detector and generate output
        result = detector.run_detector(img)
        image_with_boxes = draw_boxes(
            img,
            result["detection_boxes"],
            result["detection_classes"],
            result["detection_scores"]
        )
        # inference done for the frame, end the timer
        end_time = time.time()

        # print the time taken in console
        print(f"Inference time: {end_time-start_time}s")
        inference_timings.append(end_time-start_time)

        # show the window in the frame
        cv2.imshow("image", image_with_boxes)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def launch_camera():
    cap = cv2.VideoCapture("pedestrians.mp4")
    detector = Detector()

    # generate a window named "image"
    cv2.namedWindow("image")

    inference_timings = []

    try:
        run_camera(cap, detector, inference_timings)
    except KeyboardInterrupt:
        print("typed Ctrl+C, closing")
    finally:
        cap.release()
        avg_time = sum(inference_timings)/len(inference_timings)
        print(f"Avg Time per frame: {avg_time}s")


if __name__ == '__main__':
    launch_camera()
