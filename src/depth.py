import torch
import cv2
import numpy as np

# Load MiDaS model once at import
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def estimate_depth(frame_rgb):
    """
    Estimate depth from an RGB frame using MiDaS_small.
    Returns depth map scaled to 0-255 (uint8) with same height/width as input frame.
    """
    # Apply transform
    input_batch = transform(frame_rgb)
    # Ensure batch dimension
    if input_batch.dim() == 3:  # (C, H, W)
        input_batch = input_batch.unsqueeze(0)  # (1, C, H, W)

    # Run MiDaS
    with torch.no_grad():
        prediction = midas(input_batch)

    depth = prediction.squeeze().cpu().numpy()  # remove batch dim

    # Normalize to 0-255
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = depth.astype(np.uint8)

    # Resize depth to match original frame
    h, w = frame_rgb.shape[:2]
    depth_resized = cv2.resize(depth, (w, h))

    return depth_resized
