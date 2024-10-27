import time
import numpy as np
import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------- Track and Store Object Paths --------------------
object_paths = {}  # Store the path history of each object by ID

def load_model(weights, device):
    """Load YOLO model with FP16 precision if supported."""
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=True)
    return model, model.names

def resize_and_pad(image, stride=32):
    """Resize and pad the image to match the model's stride."""
    h, w = image.shape[:2]
    new_h = (h + stride - 1) // stride * stride
    new_w = (w + stride - 1) // stride * stride
    padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    padded_image[:h, :w, :] = image
    return padded_image

def generate_unique_color(track_id):
    """Generate a unique color for each track ID."""
    np.random.seed(int(track_id))  # Seed with track ID for consistent colors
    return tuple(np.random.randint(0, 255, 3).tolist())  # Random RGB color


def draw_object_path(frame, path, color):
    """Draw the path of the tracked object on the frame."""
    for i in range(1, len(path)):
        if path[i - 1] and path[i]:
            cv2.line(frame, path[i - 1], path[i], color, 2)

def inference(frame, model, names, tracker, line_thickness=2):
    """Perform inference, track objects, and return annotated frame with FPS."""
    start_time = time.time()

    # Resize and prepare input image
    img = resize_and_pad(frame, stride=model.stride)

    # Fix for negative strides issue by making the array contiguous
    img = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))

    # Convert to PyTorch tensor and move to device
    img = torch.from_numpy(img).to(model.device).float() / 255.0
    img = img.half() if model.fp16 else img  # Use FP16 if available
    img = img.unsqueeze(0) if img.ndimension() == 3 else img

    # Run inference and apply Non-Maximum Suppression (NMS)
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45, max_det=100)

    # Initialize annotator for drawing on the frame
    annotator = Annotator(frame, line_width=line_thickness, example=str(names))

    # Prepare detections for the tracker
    detections = []  # Initialize an empty list for detections

    # Check if predictions contain valid detections
    if len(pred) > 0 and len(pred[0]) > 0:
        detections = [
            (
                [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])],  # Bounding box [x, y, w, h]
                conf.item(),  # Confidence score
                names[int(cls)]  # Class name
            )
            for *xyxy, conf, cls in pred[0]  # Properly unpack each detection
        ]

    # Update tracker with current detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked objects and their paths
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        bbox = track.to_ltrb()  # Get bounding box (left, top, right, bottom)
        track_id, track_cls = track.track_id, track.det_class
        center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

        # Store the object's path
        if track_id not in object_paths:
            object_paths[track_id] = []  # Initialize path for new track ID
        object_paths[track_id].append(center)  # Add current position to path

        # Draw path and bounding box
        color = generate_unique_color(track_id)  # Unique color for the track ID
        draw_object_path(frame, object_paths[track_id], color)  # Draw path
        label = f'ID: {track_id} | {track_cls}'
        annotator.box_label(bbox, label, color=color)  # Draw bounding box

    # Calculate and display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotator.result()



def run(weights='yolov9-c.pt', device=0):
    """Run object detection and tracking using the webcam."""
    model, names = load_model(weights, device)
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=None, embedder_gpu=True, half=True)

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("G:/UTS/2024/Spring_2024/Image Processing/Assignment/Video-Analytics-/data_/traffic_4.mp4") 

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform inference and display the result
        annotated_frame = inference(frame, model, names, tracker)
        cv2.imshow('YOLOv9 with DeepSORT Tracking and Path Tracing', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
