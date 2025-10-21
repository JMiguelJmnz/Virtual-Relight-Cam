# capture.py
import cv2
import pyvirtualcam
import time

# List of common resolutions (descending order)
COMMON_RESOLUTIONS = [
    (3840, 2160),  # 4K
    (2560, 1440),  # QHD
    (1920, 1080),  # FHD
    (1280, 720),   # HD
    (640, 480),    # SD
]

class CaptureManager:
    def __init__(self, downscale=0.5, fps=30):
        self.cap = cv2.VideoCapture(0)
        self.downscale = downscale
        self.fps = fps

        # Auto-detect max resolution
        self.native_width, self.native_height = self.detect_max_resolution()
        print(f"Camera max detected resolution: {self.native_width}x{self.native_height}")

        # Apply downscale
        self.width = int(self.native_width * self.downscale)
        self.height = int(self.native_height * self.downscale)
        print(f"Using downscaled resolution: {self.width}x{self.height}")

        # Initialize virtual camera
        self.cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps)
        print("Virtual camera started")

    def detect_max_resolution(self):
        """Detect the highest supported resolution from COMMON_RESOLUTIONS."""
        for w, h in COMMON_RESOLUTIONS:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w == w and actual_h == h:
                return w, h
        # fallback if nothing matches
        return 640, 360

    def read_frame(self):
        start_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            return None
        # Downscale for processing
        frame_small = cv2.resize(frame, (self.width, self.height))

        end_time = time.time()
         # print(f"[Timing] read_frame() took {end_time - start_time:.3f}s")
        
        return frame_small

    def send_frame(self, frame_small):
        start_time = time.time()
        
        self.cam.send(frame_small)
        self.cam.sleep_until_next_frame()

        end_time = time.time()
        # print(f"[Timing] send_frame() took {end_time - start_time:.3f}s")

    def release(self):
        self.cap.release()
        self.cam.close()
