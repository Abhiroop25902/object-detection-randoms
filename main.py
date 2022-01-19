import time
from core.imageDrawer import draw_boxes
import cv2
# from core.imageUtils import (
#     download_and_resize_image,
#     load_img,
#     save_img,
#     show_image
# )
from core.objectDetector import Detector


def launch_camera():
    cap = cv2.VideoCapture("pedestrians.mp4")
    detector = Detector()

    # generate a window named "image"
    cv2.namedWindow("image")

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
        print(f"Inference time: {end_time-start_time} s")

        # show the window in the frame
        cv2.imshow("image", image_with_boxes)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    launch_camera()

    # downloaded_image_path = download_and_resize_image(IMAGE_URL, 1280, 856)
    # img = load_img(downloaded_image_path)

    # detector = Detector()
    # result = detector.run_detector(img)

    # image_with_boxes = draw_boxes(
    #     img,
    #     result["detection_boxes"],
    #     result["detection_classes"],
    #     result["detection_scores"]
    # )
    # show_image(image_with_boxes)
    # save_img(image_with_boxes)
