import cv2
import numpy as np
from rembg import remove, new_session
import time
import os

# Cache the ONNX session so the model is loaded only once
_cached_session = None

def remove_background(frame_small):
    global _cached_session

    total_start = time.time()

    # Step 1: Model load / session cache
    start = time.time()
    if _cached_session is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "..", "models")

        os.makedirs(model_dir, exist_ok=True)

        # Create ONNX session and prefer GPU execution
        _cached_session = new_session(
            "u2netp",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            model_dir=model_dir
        )
        print(f"[Init] Model loaded from: {model_dir}")
    print(f"[Timing] Model session setup: {time.time() - start:.3f}s")

    # Step 2: Preprocessing
    start = time.time()
    if frame_small is None or not isinstance(frame_small, np.ndarray):
        raise ValueError("Invalid frame passed to remove_background")
    
    h, w, _ = frame_small.shape
    print(f"[Input] Frame shape: {w}x{h}")
    print(f"[Timing] Preprocessing: {time.time() - start:.3f}s")

    # Step 3: Background removal (returns RGBA)
    start = time.time()
    fg = remove(frame_small, session=_cached_session)
    print(f"[Timing] inference (background removal): {time.time() - start:.3f}s")

    # Step 4: Postprocessing (convert & blend)
    start = time.time()
    fg_bgr = cv2.cvtColor(fg, cv2.COLOR_RGBA2BGR)
    alpha = fg[:, :, 3] / 255.0  # Alpha channel as mask (0.0 to 1.0)

    green_bg = np.full_like(fg_bgr, (0, 255, 0), dtype=np.uint8) # Create a pure chroma green background (adjust if needed)
    result = (fg_bgr * alpha[..., None] + green_bg * (1 - alpha[..., None])).astype(np.uint8) # Blend foreground with green background using alpha
    print(f"[Timing] Postprocessing & blend: {time.time() - start:.3f}s")
    
    # Step 5: Total
    print(f"[Timing] Total remove_background(): {time.time() -start:.3f}s\n")

    return result
