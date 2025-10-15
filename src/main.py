import cv2
from src.capture import CaptureManager
from src.background import remove_background
from src.relight import relight_curve
import threading
from queue import Queue
import time

frame_queue = Queue(maxsize=1)
processed_queue = Queue(maxsize=1)

def processing_worker():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        start = time.time()
        processed = remove_background(frame)
        mid = time.time()
        processed = relight_curve(processed)
        end = time.time()

        print(f"Background: {mid - start:.2f}s, Relight: {end - mid:.2f}s")

        if processed_queue.full():
            processed_queue.get() # drop old frame
        processed_queue.put(processed)
        print("Processing frame...")

threading.Thread(target=processing_worker, daemon=True).start()


def main():
    capture = CaptureManager(downscale=0.2)

    prev_time = time.time()
    frame_count = 0

    while True:
   
        frame = capture.read_frame()
        if frame is None:
            break

        if not frame_queue.full():
            frame_queue.put(frame)

        if not processed_queue.empty():
            capture.send_frame(processed_queue.get())

        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - prev_time)
            print(f"Approx FPS: {fps:.2f}")
            prev_time = now

        print(f"Frame queue size: {frame_queue.qsize()}, Processed queue: {processed_queue.qsize()}")

        #fg = remove_background(frame)
        #processed = relight_curve(fg, gamma=0.9, s_curve_strength=0.0)
        #capture.send_frame(processed)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
