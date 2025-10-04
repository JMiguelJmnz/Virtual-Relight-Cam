import cv2
import pyvirtualcam

def main():
    cap = cv2.VideoCapture(0)
    cam = pyvirtualcam.Camera(width=640, height=480, fps=30)
    print("Virtual camera started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: resize to match virtual cam
        frame_resized = cv2.resize(frame, (640, 480))

        # Send to virtual cam
        cam.send(frame_resized)
        cam.sleep_until_next_frame()

        # Show preview
        cv2.imshow("Passthrough Test", frame_resized)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
