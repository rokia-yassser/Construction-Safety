import cv2
import torch_directml
from ultralytics import YOLO
import cvzone
import pandas as pd
import time

# Load model and move it to DirectML device
dml_device = torch_directml.device()
model = YOLO(r"C:\Users\HP\Desktop\ML-DL-basics\safety_project\ConstructionSafety\best.pt")
model.model = model.model.to(dml_device)

# Define class names
classNames = ['boots', 'gloves', 'helmet', 'helmet on', 'no boots', 'no glove', 'no helmet', 'no vest', 'person', 'vest']

# Open video capture
cap = cv2.VideoCapture(r"C:\Users\HP\Desktop\ML-DL-basics\safety_project\ConstructionSafety\8293012-hd_1920_1080_30fps.mp4")

# Create DataFrame for detections
df = pd.DataFrame(columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

# FPS tracking
prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:
        print("Error reading frame from video stream.")
        break

    # Perform object detection
    results = model(img, stream=True)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            try:
                if 0 <= cls < len(classNames) and conf > 0.5:
                    cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=3, thickness=3)
                    df = pd.concat([df, pd.DataFrame({'Class': classNames[cls], 'Confidence': conf, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}, index=[0])], ignore_index=True)
                else:
                    print(f"Warning: Class index {cls} out of range. Using default label.")
                    cvzone.putTextRect(img, "Unknown Class", (max(0, x1), max(35, y1)), scale=1, thickness=1)
            except IndexError:
                print("Error: Index out of range. Skipping detection.")
    
    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display frame
    resized_img = cv2.resize(img, (800, 600))
    cv2.imshow("Image", resized_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save detections to Excel
try:
    df.to_excel('detections2.xlsx', index=False)
except ModuleNotFoundError:
    print("Error: 'openpyxl' module not found. Install it using 'pip install openpyxl'.")

# Release resources
cap.release()
cv2.destroyAllWindows()
