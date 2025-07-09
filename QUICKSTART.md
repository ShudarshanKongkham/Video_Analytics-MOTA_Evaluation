# 🚀 Quick Start Guide

Get up and running with Video Analytics in under 5 minutes!

## 1️⃣ One-Minute Setup

```bash
# Clone and install
git clone https://github.com/your-username/Video-Analytics.git
cd Video-Analytics
pip install -r requirements.txt

# Download a pre-trained model
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

## 2️⃣ Your First Detection

```python
# quick_demo.py
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov5s.pt')

# Run detection on webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('YOLO Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 3️⃣ Interactive Zone Analysis

```bash
# Run the zone analysis demo
cd YOLOv9
python zone_02.py
```

**How to use:**
1. **Define zones**: Left-click to add points, right-click to remove
2. **Complete polygon**: Press Enter when done
3. **Start analysis**: Press 's' to begin processing
4. **View results**: Watch real-time entry/exit counts

![Zone Demo](assets/zone_demo.gif)

## 4️⃣ Track Objects with Paths

```bash
# Run object tracking with path visualization
python YOLOv9/Track_Trace.py
```

This will show:
- ✅ Real-time object detection
- 🎯 Persistent object tracking with IDs
- 📍 Object path visualization
- 📊 Performance metrics (FPS)

## 5️⃣ Batch Video Processing

```python
# process_video.py
import cv2
from ultralytics import YOLO
from deep_sort_realtime import DeepSort

def process_video(input_path, output_path):
    model = YOLO('yolov5s.pt')
    tracker = DeepSort()
    
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect and track
        results = model(frame)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
        
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Draw results
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            bbox = track.to_ltrb()
            track_id = track.track_id
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', 
                       (int(bbox[0]), int(bbox[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()

# Process your video
process_video('input.mp4', 'output_tracked.mp4')
```

## 📊 Sample Outputs

### Detection Results
```
Detected Objects:
- person: 0.89 confidence at (150, 200, 300, 450)
- car: 0.92 confidence at (400, 300, 600, 400)
- bicycle: 0.76 confidence at (100, 250, 200, 350)

Tracking Results:
- ID 1 (person): tracked for 45 frames
- ID 2 (car): tracked for 120 frames  
- ID 3 (bicycle): tracked for 30 frames
```

### Zone Analysis Results
```
Zone 1 (Entry):
  - Entered: 15 objects
  - Exited: 2 objects
  - Currently inside: 13 objects

Zone 2 (Exit):
  - Entered: 8 objects
  - Exited: 12 objects
  - Currently inside: 3 objects
```

## 🎯 Common Use Cases

### 1. Webcam Object Detection
```bash
python YOLOv5/YoloV5_cocoDetect.py
```

### 2. Traffic Analysis
```bash
# Edit the video path in the script first
python YOLOv9/TrackAnalysis.py
```

### 3. People Counting
```bash
python YOLOv5/YOLOv5_DeepSort_MOT16_Person_Eval.py
```

### 4. Color-based Tracking
```bash
cd ColorSegmentation
python color_segmentation.py
```

## ⚙️ Configuration

### Model Selection
```python
# Fast but less accurate
model = YOLO('yolov5n.pt')  # Nano

# Balanced performance  
model = YOLO('yolov5s.pt')  # Small

# High accuracy
model = YOLO('yolov5l.pt')  # Large
```

### Tracking Parameters
```python
tracker = DeepSort(
    max_age=30,        # Keep track for 30 frames without detection
    n_init=3,          # Confirm track after 3 consecutive detections
    nn_budget=None,    # No limit on feature storage
    embedder_gpu=True  # Use GPU for feature extraction
)
```

### Detection Thresholds
```python
results = model(frame, conf=0.5)  # Confidence threshold
results = model(frame, iou=0.45)  # IoU threshold for NMS
```

## 🔧 Troubleshooting Quick Fixes

### Model Not Found
```bash
# Download missing models
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

### Low FPS
```python
# Resize frame for faster processing
frame = cv2.resize(frame, (640, 480))

# Use smaller model
model = YOLO('yolov5n.pt')  # Fastest model
```

### Memory Issues
```python
# Process every Nth frame
frame_count = 0
if frame_count % 2 == 0:  # Process every 2nd frame
    results = model(frame)
frame_count += 1
```

## 📱 Mobile/Edge Deployment

For deployment on edge devices:

```python
# Export to mobile-friendly format
model = YOLO('yolov5s.pt')
model.export(format='onnx')      # ONNX format
model.export(format='tflite')    # TensorFlow Lite
model.export(format='coreml')    # CoreML for iOS
```

## 🌐 Web Interface (Bonus)

Create a simple web interface:

```python
# app.py
import streamlit as st
import cv2
from ultralytics import YOLO

st.title("Video Analytics Dashboard")

# File upload
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    # Save uploaded file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    # Process video
    model = YOLO('yolov5s.pt')
    cap = cv2.VideoCapture("temp_video.mp4")
    
    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)
        annotated_frame = results[0].plot()
        
        stframe.image(annotated_frame, channels="BGR")
```

```bash
# Run web app
streamlit run app.py
```

## 🎓 Learning Path

1. **Beginner**: Start with basic detection (`YoloV5_cocoDetect.py`)
2. **Intermediate**: Add tracking (`DeepSort_ObjectTracking.py`)
3. **Advanced**: Implement zone analysis (`zone_02.py`)
4. **Expert**: Custom evaluation (`MOT16_evaluation.ipynb`)

## 📚 Next Steps

- 📖 Read the full [README.md](README.md) for detailed features
- 🔧 Check [INSTALLATION.md](INSTALLATION.md) for advanced setup
- 📊 Explore [Jupyter notebooks](ColorSegmentation/) for interactive analysis
- 🧪 Run MOT16 evaluation for performance benchmarking

## 💡 Tips for Better Results

1. **Good lighting**: Ensure adequate lighting for better detection
2. **Stable camera**: Minimize camera shake for better tracking
3. **Appropriate resolution**: Balance between quality and speed
4. **Clean background**: Reduce background clutter for better accuracy

---

**🎉 Congratulations!** You're now ready to explore advanced video analytics with YOLO and DeepSORT!

Need help? Check our [Issues](https://github.com/your-username/Video-Analytics/issues) or [Discussions](https://github.com/your-username/Video-Analytics/discussions).
