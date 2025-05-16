import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
import pandas as pd
from collections import deque
from scipy.spatial import distance

# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

def detect_persons(frame, threshold=0.8):
    img_tensor = F.to_tensor(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    persons = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if label == 1 and score >= threshold:
            persons.append(box.cpu().numpy())
    return persons

def find_crowd_groups(person_boxes, distance_threshold=150):
    if len(person_boxes) < 3:
        return []

    centers = []
    for box in person_boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))

    groups = []
    visited = set()

    for i in range(len(centers)):
        if i in visited:
            continue
        group = [i]
        for j in range(len(centers)):
            if i != j and j not in visited:
                d = distance.euclidean(centers[i], centers[j])
                if d < distance_threshold:
                    group.append(j)
        if len(group) >= 3:
            visited.update(group)
            groups.append(tuple(sorted(group)))  # store indices of persons in the group

    return groups

def main():
    cap = cv2.VideoCapture("input_video.mp4")
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_history = deque(maxlen=10)  # Keep groups from last 10 frames
    crowd_log = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons = detect_persons(frame)
        print(f"Frame {frame_num}: {len(persons)} persons detected.")

        frame_groups = find_crowd_groups(persons)
        frame_history.append(frame_groups)

        # Check persistence of groups across last 10 frames
        persistent_crowds = {}
        for group in frame_groups:
            count = sum([group in f for f in frame_history])
            if count == 10:  # group persisted for 10 consecutive frames
                persistent_crowds[group] = len(group)

        for group, count in persistent_crowds.items():
            if not any(log[1] == count and log[0] == frame_num for log in crowd_log):
                crowd_log.append((frame_num, count))

        frame_num += 1

    cap.release()
    df = pd.DataFrame(crowd_log, columns=["Frame Number", "Person Count in Crowd"])
    df.to_csv("crowd_detection_log.csv", index=False)
    print("Crowd detection complete. Results saved to 'crowd_detection_log.csv'.")

if __name__ == "__main__":
    main()
