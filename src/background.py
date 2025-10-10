import cv2
import numpy as np
from rembg import remove

def remove_background(frame):
    # Run background removal (returns RGBA)
    fg = remove(frame, model_dir="E:/Python/Virtual Relight Cam/models")

    # Convert RGBA → BGR
    fg_bgr = cv2.cvtColor(fg, cv2.COLOR_RGBA2BGR)
    alpha = fg[:, :, 3] / 255.0  # Alpha channel as mask (0.0 to 1.0)

    # Create a pure chroma green background (adjust if needed)
    green_bg = np.full_like(fg_bgr, (0, 255, 0), dtype=np.uint8)

    # Blend foreground with green background using alpha
    result = (fg_bgr * alpha[..., None] + green_bg * (1 - alpha[..., None])).astype(np.uint8)

    return result
