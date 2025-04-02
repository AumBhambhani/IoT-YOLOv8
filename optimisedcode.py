import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("../models/yolov8n.pt")  # Ensure the correct path

# Read video
video_path = "/home/student/IoT-YOLOv8/data/test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define people class index (COCO dataset: 'person' is class 0)
PERSON_CLASS_ID = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    # Run YOLO detection
    results = model(frame)

    person_count = 0  # Initialize person count for this frame

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class ID
            confidence = box.conf[0].item()  # Get confidence score

            # Filter: Only detect people (class ID 0) with confidence > 0.7
            if class_id == PERSON_CLASS_ID and confidence > 0.7:
                person_count += 1  # Increase count
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box
                cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display person count on screen
    cv2.putText(frame, f"Total People: {person_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("YOLO People Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
