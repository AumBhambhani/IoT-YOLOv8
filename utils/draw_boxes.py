import cv2

def draw_boxes(frame, results, conf_threshold=0.7):
    """
    Draw bounding boxes and labels on the frame only for detected people 
    with confidence above the threshold.
    
    Args:
        frame: The image frame (numpy array).
        results: YOLO detection results.
        conf_threshold: Minimum confidence score to consider a detection.
    """
    for box in results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = box.tolist()
        class_id = int(class_id)  # Convert to integer

        # Only draw for people (class ID 0 in COCO dataset) with confidence > 0.7
        if class_id == 0 and confidence > conf_threshold:
            label = f"Person: {confidence:.2f}"
            color = (0, 255, 0)  # Green for people

            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - h - 5), (int(x1) + w, int(y1)), color, -1)

            # Put label text
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame
