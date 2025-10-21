import cv2
from src.capture import CaptureManager
from src.background import remove_background
from src.relight import relight_curve
import threading
from queue import Queue
import time

class DoubleBuffer:
    def __init__(self):
        self.input_buffer = None
        self.output_buffer = None
        self.lock_in = threading.Lock()
        self.lock_out = threading.Lock()

    def set_input(self, frame_small):
        with self.lock_in:
            self.input_buffer = frame_small
    
    def get_input(self):
        with self.lock_in:
            frame_small = self.input_buffer
            self.input_buffer = None
            return frame_small
        
    def set_output(self, frame_small):
        with self.lock_out:
            self.output_buffer = frame_small

    def get_output(self):
        with self.lock_out:
            return self.output_buffer

def processing_thread(buffers):
    while True:
        frame_small = buffers.get_input()
        if frame_small is None:
            time.sleep(0.005)
            continue

        start = time.time()
        processed = remove_background(frame_small)
        processed = relight_curve(processed)
        end = time.time()

        print(f"[Processing] Total: {end - start:.3f}s")
        buffers.set_output(processed)

def main():
    capture = CaptureManager(downscale=0.5)
    buffers = DoubleBuffer()

    threading.Thread(target=processing_thread, args=(buffers,), daemon=True).start()

    prev_time = time.time()
    frame_count = 0
    fps_display = 0.0

    while True:
        frame_small = capture.read_frame()
        if frame_small is None:
            break

        buffers.set_input(frame_small)

        output_frame = buffers.get_output()

        if output_frame is not None:
            # FPS counter
            frame_count += 1
            now = time.time()
            if frame_count % 10 == 0:
                fps_display = 10 / (now - prev_time)
                prev_time = now

            cv2.putText(
                    output_frame,
                    f"{fps_display:.1f} FPS",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )
            
            capture.send_frame(output_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
