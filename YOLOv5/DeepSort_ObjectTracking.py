import cv2
import numpy as np
import time
import torch
import os

from deep_sort_realtime.deepsort_tracker import DeepSort

class YoloDetector():
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 4 
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame_resized = cv2.resize(frame, (width, height))
        results = self.model(frame_resized)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, confidence=0.3):
        labels, cord = results
        detections = []

        height, width = frame.shape[:2]
        for i in range(len(labels)):
            row = cord[i]
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0] * width), int(row[1] * height), int(row[2] * width), int(row[3] * height)
                
                if self.class_to_label(labels[i]) == 'person':
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'person'))
        
        return frame, detections

# Initialize YOLO detector
detector = YoloDetector(model_name=None)

# Initialize DeepSORT tracker
object_tracker = DeepSort(max_age=5,
                n_init=2,
                nms_max_overlap=1.0,
                max_cosine_distance=0.7,
                nn_budget=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True)

# Open video capture
cap = cv2.VideoCapture("walking.mp4")
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set up environment variable for compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    start = time.perf_counter()
    
    # Perform detection
    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(results, img, confidence=0.5)

    # Update tracks
    tracks = object_tracker.update_tracks(detections, frame=img)

    # Draw boxes and IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        track_age = track.age
        track_class = track.det_class
        bbox = track.to_ltrb()  # Get bounding box in [left, top, right, bottom] format
        
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, f"{track_class}", (int(bbox[0]), int(bbox[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, f"ID: {track_id}", (int(bbox[0]), int(bbox[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"AGE: {track_age}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
