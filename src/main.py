import cv2
from src.capture import CaptureManager
from src.background import remove_background
from src.relight import relight_curve
import threading
from queue import Queue

frame_queue = Queue(maxsize=1)
processed_queue = Queue(maxsize=1)

def main():
    capture = CaptureManager(downscale=0.2)

    while True:
        frame = capture.read_frame()
        if frame is None:
            break

        fg = remove_background(frame)
        processed = relight_curve(fg, gamma=0.9, s_curve_strength=0.3)
        capture.send_frame(processed)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
