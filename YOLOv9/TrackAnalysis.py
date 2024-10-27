import time
import numpy as np
import torch
import cv2
from models.common import DetectMultiBackend  # Model wrapper to support multiple backends
from utils.general import non_max_suppression, scale_boxes  # Utilities for detection
from utils.plots import Annotator  # Annotator for drawing boxes and labels
from utils.torch_utils import select_device  # Device selector (CPU/GPU)
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT tracker for object tracking

# Object tracking paths and entry/exit statistics
object_paths = {}  # Stores the paths of tracked objects
object_directions = {"entered": 0, "exited": 0}  # Count of objects entering and exiting

# Define three line coordinates for tracking movement across predefined zones
ENTRY_LINE_Y = 310  # Entry line
MID_LINE_Y = 320    # Mid-line (can confirm movement direction)
EXIT_LINE_Y = 350   # Exit line

### 1. Load Model
def load_model(weights, device):
    """Load the YOLOv9 model and set precision if supported."""
    device = select_device(device)  # Select CPU or GPU for inference
    model = DetectMultiBackend(weights, device=device, fp16=True)  # Load model with FP16 precision if available
    return model, model.names  # Return model and class names

### 2. Generate Unique Colors
def generate_unique_color(track_id):
    """Generate a unique color for each tracked object."""
    np.random.seed(int(track_id))  # Use track ID as a seed for consistent colors
    return tuple(np.random.randint(0, 255, 3).tolist())  # Return RGB color tuple

### 3. Draw Paths
def draw_object_path(frame, path, color):
    """Draw the path of a tracked object on the frame."""
    for i in range(1, len(path)):
        if path[i - 1] and path[i]:  # Ensure both points are valid
            cv2.line(frame, path[i - 1], path[i], color, 2)  # Draw line between consecutive points

### 4. Analyze Crossings
def analyze_crossing(path):
    """Determine if the object crossed the entry, mid, or exit lines."""
    if len(path) < 2:
        return  # Need at least two points to detect movement

    # Compare previous and current y-coordinates to detect crossing
    prev_y, curr_y = path[-2][1], path[-1][1]

    # Increment counters based on crossing direction
    if prev_y < ENTRY_LINE_Y and curr_y >= ENTRY_LINE_Y:
        object_directions["entered"] += 1
    elif prev_y > EXIT_LINE_Y and curr_y <= EXIT_LINE_Y:
        object_directions["exited"] += 1

### 5. Perform Inference and Tracking
def inference(frame, model, names, tracker, path_frame, line_thickness=2):
    """Perform inference, update tracker, and return annotated frames."""
    start_time = time.time()  # Start time for FPS calculation

    # Convert frame to model input format (BGR to RGB, HWC to CHW)
    img = np.ascontiguousarray(frame[..., ::-1].transpose(2, 0, 1))
    img = torch.from_numpy(img).to(model.device).float() / 255.0  # Normalize

    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension if missing

    img = img.half() if model.fp16 else img  # Convert to half precision if supported

    # Run inference and apply Non-Max Suppression (NMS) to filter predictions
    with torch.no_grad():
        predictions = model(img)
        pred = non_max_suppression(predictions[0], conf_thres=0.5, iou_thres=0.45, max_det=100)

    annotator = Annotator(frame, line_width=line_thickness, example=str(names))  # Annotate detections

    # Prepare detections for DeepSORT tracker
    detections = [
        (
            [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])],  # Bounding box
            conf.item(),  # Confidence score
            names[int(cls)]  # Class name
        )
        for *xyxy, conf, cls in pred[0]
    ] if len(pred[0]) > 0 else []

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Process each confirmed track
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue  # Skip unconfirmed or outdated tracks

        bbox = track.to_ltrb()  # Get bounding box coordinates
        track_id = track.track_id  # Track ID
        center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))  # Compute object's center

        # Store the object's path
        if track_id not in object_paths:
            object_paths[track_id] = []
        object_paths[track_id].append(center)

        # Analyze crossing based on path
        analyze_crossing(object_paths[track_id])

        # Draw path and annotate bounding box with track ID
        color = generate_unique_color(track_id)
        draw_object_path(path_frame, object_paths[track_id], color)
        annotator.box_label(bbox, f'ID: {track_id}', color=color)

    # Draw lines on both frames
    for frame_to_draw in [frame, path_frame]:
        cv2.line(frame_to_draw, (0, ENTRY_LINE_Y), (frame.shape[1], ENTRY_LINE_Y), (0, 255, 0), 2)
        cv2.line(frame_to_draw, (0, MID_LINE_Y), (frame.shape[1], MID_LINE_Y), (255, 255, 0), 2)  # Yellow mid-line
        cv2.line(frame_to_draw, (0, EXIT_LINE_Y), (frame.shape[1], EXIT_LINE_Y), (0, 0, 255), 2)

    # Calculate and display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotator.result(), path_frame

### 6. Main Run Function
def run(weights='yolov9-c.pt', device=0):
    """Run the YOLO model with object tracking and visualizations."""
    model, names = load_model(weights, device)  # Load model
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=None, embedder_gpu=True, half=True)  # Initialize tracker

    cap = cv2.VideoCapture("G:/UTS/2024/Spring_2024/Image Processing/Assignment/Video-Analytics-/data_/traffic_1.mp4") 


    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, sample_frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        return

    path_frame = np.zeros_like(sample_frame)  # Initialize path frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)

        # Perform inference and get annotated frames
        annotated_frame, path_frame = inference(frame, model, names, tracker, path_frame)

        # Display real-time detection and path analysis windows
        cv2.imshow('YOLOv9 Real-Time Detection', annotated_frame)
        cv2.imshow('Object Path Analysis', path_frame)

        # Update and display counts without overlay
        cv2.rectangle(path_frame, (0, 0), (400, 60), (0, 0, 0), -1)  # Clear text area
        cv2.putText(path_frame, f'Entered: {object_directions["entered"]}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(path_frame, f'Exited: {object_directions["exited"]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows


if __name__ == "__main__":
    run()
