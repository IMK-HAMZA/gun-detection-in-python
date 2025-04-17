from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 model (general object detection)
model = YOLO("yolov8n.pt")  # 'n' stands for nano - very lightweight

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.5)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
