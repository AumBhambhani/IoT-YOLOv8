from ultralytics import YOLO
import shutil

# Download the model
model = YOLO("yolov8n.pt")

# Move it to a custom path
shutil.move("yolov8n.pt", "C:/Users/aum/Documents/SDI/yolo/models/yolov8n.pt")
