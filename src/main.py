import cv2
from src.capture import CaptureManager
from src.background import remove_background

def main():
    capture = CaptureManager(downscale=0.5)

    while True:
        frame = capture.read_frame()
        if frame is None:
            break

        processed = remove_background(frame)
        capture.send_frame(processed)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    capture.release()

if __name__ == "__main__":
    main()
