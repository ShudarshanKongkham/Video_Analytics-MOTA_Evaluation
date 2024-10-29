import time
import numpy as np
import torch
import cv2
import os
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from deep_sort_realtime.deepsort_tracker import DeepSort

def load_model(weights, device):
    """Load the YOLO model with specified weights and device."""
    device = select_device(device)  # Select GPU or CPU
    model = DetectMultiBackend(weights, device=device, fp16=True)  # Enable FP16 for GPU if supported
    return model, model.names

def resize_and_pad(image, stride=32):
    """Resize and pad the image to be compatible with the model's stride."""
    # print("Before Padding", image.shape)

    h, w = image.shape[:2]
    new_h = (h + stride - 1) // stride * stride
    new_w = (w + stride - 1) // stride * stride
    padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    padded_image[:h, :w, :] = image
    # print("After Padding",padded_image.shape)
    return padded_image

def inference(image, model, names, deepsort, line_thickness=2):
    """Perform inference and return annotated image with tracking data."""
    start_time = time.time()  # Track inference time for FPS calculation

    # Resize and pad image to be stride-compatible
    padded_image = resize_and_pad(image, stride=model.stride)

    # Prepare image for inference
    img = padded_image[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, 3xHxW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(model.device).float() / 255.0
    if model.fp16:  # Use half precision if available
        img = img.half()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    # Run inference
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, max_det=100)

    # Initialize annotator
    im0 = image.copy()
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

    # Process person-only detections
    person_index = next((k for k, v in names.items() if v == "person"), None)

    detections = []
    if person_index is not None and len(pred[0]):
        det = pred[0]
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            if int(cls) == person_index:  # Filter for person detections only
                x1, y1, x2, y2 = map(int, xyxy)
                cls_name = names[int(cls)]  # Class name from dictionary
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), cls_name))

    # Update tracker with person-only detections
    tracks = deepsort.update_tracks(detections, frame=im0)

    # Initialize list to store tracking data for this frame
    tracking_data = []

    # Annotate tracked objects and collect tracking data
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        track_cls = track.det_class

        bbox = track.to_ltrb()  # left, top, right, bottom

        # Assign a unique color to the person class
        cls_color = colors(0, True)  # Assuming "person" is class index 0

        # Clean and readable annotations
        label = f'ID: {track_id} | {track_cls}'
        annotator.box_label(bbox, label, color=cls_color)

        # Extract bounding box coordinates
        bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
        bbox_width = bbox_right - bbox_left
        bbox_height = bbox_bottom - bbox_top

        # Get detection confidence from the track
        confidence = track.det_conf  # Ensure DeepSORT returns this attribute

        # Append data to tracking_data
        tracking_data.append([track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence])

    # Calculate FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(im0, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return annotator.result(), tracking_data



def process_images_from_folder(weights, device, img_folder):
    """Process images from a folder, save results to a video, and create res.txt."""
    model, names = load_model(weights, device)

    # Get the parent directory to create the 'yolov9' folder inside it
    parent_dir = os.path.dirname(img_folder)
    output_dir = os.path.join(parent_dir, "yolov9")
    os.makedirs(output_dir, exist_ok=True)  # Create 'yolov9' folder if not exists

    # Set the output video name based on the parent folder name
    video_name = os.path.basename(parent_dir) + ".avi"
    output_path = os.path.join(output_dir, video_name)

    # Get image file paths
    img_files = sorted(os.listdir(img_folder))
    img_paths = [os.path.join(img_folder, img_file) for img_file in img_files]

    if len(img_paths) == 0:
        print(f"No images found in {img_folder}")
        return

    # Load the first image to get frame dimensions
    sample_img = cv2.imread(img_paths[0])
    if sample_img is None:
        print(f"Error: Could not read {img_paths[0]}.")
        return

    height, width = sample_img.shape[:2]
    fps = 25  # Assuming 25 FPS for the output video

    # Initialize video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize DeepSORT tracker
    deepsort = DeepSort(
        max_age=30, n_init=3, nms_max_overlap=1.0,
        max_cosine_distance=0.7, nn_budget=None,
        embedder_gpu=True, half=True
    )

    # Initialize tracking results list
    tracking_results = []

    # Process each image
    for frame_number, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Error: Could not read {img_path}. Skipping.")
            continue

        print(f"Processing Frame {frame_number}: {img_path}")

        # Perform inference and get annotated frame and tracking data
        annotated_frame, tracking_data = inference(frame, model, names, deepsort)

        # Collect tracking results
        for data in tracking_data:
            track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence = data
            # Prepare the line in required format
            res_line = [
                frame_number,  # Frame number starting from 1
                track_id,
                bbox_left,
                bbox_top,
                bbox_width,
                bbox_height,
                confidence,
                -1, -1, -1  # Placeholders for world coordinates
            ]
            tracking_results.append(res_line)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow('YOLOv9 with DeepSORT', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    out.release()
    cv2.destroyAllWindows()

    # Write tracking results to res.txt
    res_txt_path = os.path.join(output_dir, 'res.txt')
    with open(res_txt_path, 'w') as f:
        for res_line in tracking_results:
            line_str = ','.join(map(str, res_line))
            f.write(line_str + '\n')

    print(f"Tracking results saved to {res_txt_path}")


if __name__ == "__main__":
    weights = 'yolov9-c.pt'  # Path to model weights
    device = 0  # Use GPU (0) or CPU ('cpu')
    MOT_sequences = ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09" ,"MOT16-10" ,"MOT16-11", "MOT16-13"]
    for MOT_folder in MOT_sequences:
        print(f"Generating for {MOT_folder} : ")
        img_folder = f"G:/UTS/2024/Spring_2024/Image Processing/Assignment/Video-Analytics-/MOT_Evaluation/MOT16/train/{MOT_folder}/img1"
        process_images_from_folder(weights, device, img_folder)
