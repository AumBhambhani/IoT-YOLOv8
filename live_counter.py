import cv2
from ultralytics import YOLO
from utils.draw_boxes import draw_boxes

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
conf_threshold = 0.7  # Set confidence threshold to 0.7

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Count people with confidence > 0.7
    people_count = sum(1 for obj in results[0].boxes.data if int(obj[-1]) == 0 and obj[-2] > conf_threshold)

    # Draw bounding boxes only for high-confidence people
    frame = draw_boxes(frame, results, conf_threshold)

    # Display count
    cv2.putText(frame, f"People: {people_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
