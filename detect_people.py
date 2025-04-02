import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("../models/yolov8n.pt")  # Ensure correct path

# Open video file or webcam (0 for default webcam)
video_path = "/home/student/IoT-YOLOv8/data/test_video.mp4"  # Change to 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run YOLO detection
    results = model(frame)

    # Draw results
    for result in results:
        annotated_frame = result.plot()  # Draw bounding boxes

    # Display output
    cv2.imshow("YOLO People Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
