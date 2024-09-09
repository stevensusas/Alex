import cv2
import numpy as np

def detect_circles():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve circle detection
        gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Detect circles in the frame using HoughCircles with higher thresholds
        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=50,
            param1=150,  # Higher threshold for Canny edge detector (more confident edge detection)
            param2= 30,   # Accumulator threshold for circle centers (higher = more confident circles)
            minRadius=5,  # Adjust minimum radius to ignore smaller circles
            maxRadius=200  # Adjust maximum radius to ignore larger circles
        )

        # Get global center of the frame
        frame_height, frame_width = frame.shape[:2]
        global_center = (frame_width // 2, frame_height // 2)
        cv2.circle(frame, global_center, 5, (255, 0, 0), -1)  # Blue circle at the center

        # If some circles are detected, draw them on the frame
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Draw the circle in the output frame
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                # Draw the center of the circle
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dot for center

                # Print circle center and global center
                print(f"Global Center: {global_center}, Circle Center: ({x}, {y})")

        # Show the frame
        cv2.imshow("Circle Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_circles()
