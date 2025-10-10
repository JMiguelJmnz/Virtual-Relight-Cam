import cv2
import numpy as np

def relight_curve(frame, gamma=1.0, s_curve_strength=0.0):

    # Normalize to 0-1 float
    frame_f = frame.astype(np.float32) / 255.0

    if gamma != 1.0:
        frame_f = np.power(frame_f, gamma)

    if s_curve_strength != 0.0:
        a = 0.5 - s_curve_strength * 0.5 # Midtones
        frame_f = 1/(1+np.exp(-12 * (frame_f - a))) # Sigmoid curve
        frame_f = np.clip(frame_f, 0,1)

    # Convert back to uint8
    
    return (frame_f * 255).astype(np.uint8)
