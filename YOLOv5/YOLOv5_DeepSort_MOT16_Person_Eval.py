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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
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
            if row[4] >= confidence and self.class_to_label(labels[i]) == 'person':  # Filter for 'person'
                x1, y1, x2, y2 = int(row[0] * width), int(row[1] * height), int(row[2] * width), int(row[3] * height)
                detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'person'))  # Only 'person'

        return frame, detections

def process_images_from_folder(device, img_folder):
    """Process images from a folder and save results as video and res.txt."""
    detector = YoloDetector(model_name=None)

    # Initialize DeepSORT tracker
    object_tracker = DeepSort(
        max_age=5,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.7,
        nn_budget=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True
    )

    # Prepare output paths
    output_folder = os.path.join(os.path.dirname(img_folder), "YOLOv5")
    os.makedirs(output_folder, exist_ok=True)
    video_name = os.path.basename(os.path.dirname(img_folder)) + ".avi"
    output_path = os.path.join(output_folder, video_name)
    res_file_path = os.path.join(output_folder, "res.txt")

    # Initialize video writer and res.txt
    img_files = sorted(os.listdir(img_folder))
    img_paths = [os.path.join(img_folder, img_file) for img_file in img_files]
    if not img_paths:
        print(f"No images found in {img_folder}")
        return

    sample_img = cv2.imread(img_paths[0])
    if sample_img is None:
        print(f"Error: Could not read {img_paths[0]}.")
        return

    height, width = sample_img.shape[:2]
    fps = 25  # Assuming 25 FPS
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    res_file = open(res_file_path, "w")

    # Process each image
    for frame_id, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}. Skipping.")
            continue

        start = time.perf_counter()
        results = detector.score_frame(img)
        img, detections = detector.plot_boxes(results, img, confidence=0.5)
        tracks = object_tracker.update_tracks(detections, frame=img)

        # Draw boxes, IDs, and write to res.txt
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            res_file.write(f"{frame_id},{track_id},{x1},{y1},{x2 - x1},{y2 - y1},1.0,1,1\n")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, f"ID: {track_id}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        totalTime = time.perf_counter() - start
        fps = 1 / totalTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        out.write(img)
        cv2.imshow('YOLOv5 with DeepSORT', img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    out.release()
    res_file.close()
    cv2.destroyAllWindows()

    print(f"Output video saved at: {output_path}")
    print(f"Tracking results saved at: {res_file_path}")

# Run the function for multiple sequences
if __name__ == "__main__":

    device = 0  # Use GPU (0) or CPU ('cpu')
    MOT_sequences = ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]

    for MOT_folder in MOT_sequences:
        print(f"Generating for {MOT_folder}...")
        img_folder = f"G:/UTS/2024/Spring_2024/Image Processing/Assignment/Video-Analytics-/MOT_Evaluation/MOT16/train/{MOT_folder}/img1"
        process_images_from_folder( device, img_folder)
