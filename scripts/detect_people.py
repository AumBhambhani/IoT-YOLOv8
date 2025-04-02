import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("../models/yolov8n.pt")  # Ensure correct path

# Read image
image_path = "../data/test_image.jpg"
image = cv2.imread(image_path)

# Run YOLO detection
results = model(image)

# Draw results
results.show()
