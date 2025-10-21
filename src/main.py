import cv2
from src.capture import CaptureManager
from src.background import remove_background
from src.relight import relight_curve
import threading
from queue import Queue
import time

# Queues for threading
frame_queue = Queue(maxsize=2)
processed_queue = Queue(maxsize=1)

def processing_worker():
    while True:
        frame_small = frame_queue.get()
        if frame_small is None:
            break

        start = time.time()
        processed = remove_background(frame_small)
        mid = time.time()
        processed = relight_curve(processed)
        end = time.time()

        print(f"[Timing] Background: {mid - start:.2f}s, Relight: {end - mid:.2f}s")

        if processed_queue.full():
            processed_queue.get_nowait()
        processed_queue.put(processed)
        print("[Thread] Frame processed and queued")

threading.Thread(target=processing_worker, daemon=True).start()


def main():
    capture = CaptureManager(downscale=0.5)

    prev_time = time.time()
    frame_count = 0
    fps_display = 0.0

    while True:
   
        frame_small = capture.read_frame()
        if frame_small is None:
            break

        if not frame_queue.full():
            frame_queue.put(frame_small)

        if not processed_queue.empty():
            processed = processed_queue.get()
            

            frame_count += 1
            now = time.time()
            if frame_count % 10 == 0:
                fps_display = 10 / (now - prev_time)
                prev_time = now

            cv2.putText(
                    processed,
                    f"{fps_display:.1f} FPS",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )
            
            capture.send_frame(processed)

        print(f"[Queue] Input: {frame_queue.qsize()}, Output: {processed_queue.qsize()}")

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
