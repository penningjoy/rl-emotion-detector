from fer import FER
from fer.utils import draw_annotations
import cv2

def detect():
    detector = FER()
    cap = cv2.VideoCapture('emotions.mp4') # a video with discernible facial expressions

    if not cap.isOpened():
        print("cannot open video")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive any more frame. Video might be over.. Exiting!")
            break

        frame = cv2.flip(frame, 1)
        emotions = detector.detect_emotions(frame)
        if emotions:
            frame = draw_annotations(frame, emotions)

        cv2.imshow("frame", frame)

        #  Hit 'q' to quit
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()
