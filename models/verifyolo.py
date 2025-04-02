from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/aum/Documents/SDI/yolo/models/yolov8n.pt")

# Print model summary
model.info()
