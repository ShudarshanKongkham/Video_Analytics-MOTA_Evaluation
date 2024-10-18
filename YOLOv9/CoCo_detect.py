import numpy as np
import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

def load_model(weights, device, imgsz):
    """Load the YOLO model with the specified weights and device."""
    device = select_device(device)  # Select GPU or CPU
    model = DetectMultiBackend(weights, device=device)  # Load the model
    imgsz = check_img_size(imgsz, s=model.stride)  # Adjust image size
    model.warmup(imgsz=(1, 3, *imgsz))  # Warmup for better performance

    return model, model.names

def inference(image, model, names, line_thickness=3):
    """Perform inference on the input image and return predictions and annotated image."""
    # Prepare image: BGR to RGB and rearrange axes
    image = np.expand_dims(image, axis=0)
    image = image[..., ::-1].transpose((0, 3, 1, 2))  # Rearrange axes (batch, channels, height, width)
    im0 = np.squeeze(image, axis=0).transpose(1, 2, 0)  # Reconstruct the original frame

    # Use `.copy()` to avoid negative stride issues
    image = torch.from_numpy(image.copy()).to(model.device).float() / 255.0
    if len(image.shape) == 3:
        image = image[None]  # Add batch dimension

    # Run inference and apply Non-Maximum Suppression (NMS)
    pred = model(image)[0][1]
    pred = non_max_suppression(pred, conf_thres=0.70, iou_thres=0.45, max_det=1000)

    # Annotate the frame with predictions
    annotator = Annotator(np.ascontiguousarray(im0), line_width=line_thickness, example=str(names))
    predicted_classes = []

    # Process predictions
    if len(pred[0]):
        det = pred[0]
        det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(int(cls), True))
            predicted_classes.append((names[int(cls)], conf.item()))

    return predicted_classes, annotator.result()

def run(weights, device, imgsz):
    """Run real-time object detection using the webcam."""
    model, names = load_model(weights, device, imgsz)
    cap = cv2.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.resize(frame, imgsz)  # Resize to match input size
        predicted_classes, annotated_frame = inference(frame, model, names)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the annotated frame
        cv2.imshow('YOLOv9 Object Detection', annotated_frame)

        # Exit the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    weights = 'yolov9-c.pt'  # Path to model weights
    device = 0  # Use GPU (0) or CPU ('cpu')
    imgsz = (640, 640)  # Model input size
    run(weights, device, imgsz)
