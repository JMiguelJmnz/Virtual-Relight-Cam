import cv2
from rembg import remove

def remove_background(frame):
    fg = remove(frame, model_dir="E:/Python/Virtual Relight Cam/models")  # returns RGBA
    fg_bgr = cv2.cvtColor(fg, cv2.COLOR_RGBA2BGR)  # convert to 3 channels
    return fg_bgr
