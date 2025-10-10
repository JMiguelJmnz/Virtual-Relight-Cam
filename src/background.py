import cv2
from rembg import remove

def remove_background(frame):
    fg = remove(frame, model_dir="E:/Python/Virtual Relight Cam/models")  # returns RGBA
    return fg
