import cv2
import pandas as pd
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time

# Initializations
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

model1 = YOLO('yolov8m.pt')       # General detector
model = YOLO('best.pt')           # PPE detector

# Ask user for PPE to check
input_ppe = input("Enter PPE to check for (comma separated: helmet, vest, mask): ").lower().strip().split(",")
input_ppe = [p.strip() for p in input_ppe if p.strip()]
print(f"Checking for: {input_ppe}")

cv2.namedWindow('RGB')

cap = cv2.VideoCapture("h")
if not cap.isOpened():
    print("Error: Unable to open video stream")
    exit()

count = 0
worker_results_dict = {}
size = (1020, 500)
stime = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    count += 1
    frame = cv2.resize(frame, size)

    if count % 12 == 0 or count == 1:
        results = model.predict(frame)
    else:
        continue  # skip frames to keep it fast

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        class_id = int(row[5])
        conf = row[4]
        if conf > 0.8 and class_id == 5:  # Person
            x1, y1, x2, y2 = map(int, row[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            worker_frame = frame[y1:y2, x1:x2, :]
            imgRGB = cv2.cvtColor(worker_frame, cv2.COLOR_BGR2RGB)
            resu = pose.process(imgRGB)

            if count % 12 == 0 or count == 1:
                worker_results = model.predict(worker_frame)
                worker_results_dict[(x1, y1)] = worker_results
            else:
                worker_results = worker_results_dict.get((x1, y1), None)
            if worker_results is None:
                continue

            worker_a = worker_results[0].boxes.data
            worker_px = pd.DataFrame(worker_a).astype("float")

            ppe_status = {ppe: False for ppe in input_ppe}

            if resu.pose_landmarks is not None:
                for id, lm in enumerate(resu.pose_landmarks.landmark):
                    if id in [4, 12, 9, 1, 11, 24, 23, 10, 0]:
                        h, w, c = worker_frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(worker_frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                        for _, row in worker_px.iterrows():
                            class_id = int(row[5])
                            class_map = {0: "helmet", 7: "vest", 1: "mask"}
                            if class_id in class_map and class_map[class_id] in input_ppe:
                                if row[0] <= cx <= row[2] and row[1] <= cy <= row[3]:
                                    ppe_status[class_map[class_id]] = True

            if all(ppe_status.get(ppe, False) for ppe in input_ppe):
                label = "yes"
                color = (0, 255, 0)
            else:
                label = "no "
                for ppe in input_ppe:
                    if not ppe_status.get(ppe, False):
                        label += ppe + " "
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27 or count > 500:
        break

cap.release()
cv2.destroyAllWindows()
etime = time.time()
print("Elapsed time: {:.2f} seconds".format(etime - stime))
