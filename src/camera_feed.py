import cv2
import torch
import numpy as np
import pyvirtualcam
from rembg import remove
from torchvision.transforms import Compose, Resize, ToTensor

# Load MiDaS for depth
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

cap = cv2.VideoCapture(0)

with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    print("Virtual camera started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Background removal
        fg = remove(frame)  # returns RGBA image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Depth estimation
        input_batch = transform(frame_rgb).unsqueeze(0)
        with torch.no_grad():
            prediction = midas(input_batch)
            depth = prediction.squeeze().cpu().numpy()
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth = depth.astype(np.uint8)

        # Show both (side by side for testing)
        combined = np.hstack((fg[:, :, :3], cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)))

        # Send to virtual cam
        cam.send(combined)
        cam.sleep_until_next_frame()

        # Also preview
        cv2.imshow("Relight Prototype", combined)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
