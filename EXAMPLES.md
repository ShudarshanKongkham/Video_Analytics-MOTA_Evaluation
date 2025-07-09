# 📊 Examples and Demos

This directory contains practical examples demonstrating various capabilities of the Video Analytics framework.

## 📁 Directory Structure

```
examples/
├── 🚀 basic_detection.py           # Simple object detection
├── 🎯 tracking_demo.py             # Object tracking example
├── 📍 zone_analysis_demo.py        # Zone-based analytics
├── 🎨 color_tracking_demo.py       # Color-based tracking
├── 📊 performance_benchmark.py     # Performance testing
├── 🎬 batch_processing.py          # Batch video processing
├── 📱 webcam_demo.py               # Real-time webcam demo
├── 🌐 web_interface.py             # Streamlit web app
├── 📈 evaluation_demo.py           # MOT evaluation example
└── 🔧 custom_model_demo.py         # Custom model training
```

## 🚀 Basic Object Detection

### Simple Detection Example

```python
# examples/basic_detection.py
"""
Basic object detection using YOLOv5
Detects objects in a single image or video frame
"""

import cv2
from ultralytics import YOLO
import numpy as np

def detect_objects_image(image_path, model_path='yolov5s.pt'):
    """Detect objects in a single image"""
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    image = cv2.imread(image_path)
    
    # Run detection
    results = model(image)
    
    # Process results
    for r in results:
        # Get bounding boxes
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # Extract coordinates and info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(image, label, (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def detect_objects_video(video_path, output_path=None, model_path='yolov5s.pt'):
    """Detect objects in video"""
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # Video writer setup (if output path provided)
    if output_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run detection
        results = model(frame)
        
        # Count detections
        detections_this_frame = 0
        for r in results:
            if r.boxes is not None:
                detections_this_frame += len(r.boxes)
        
        total_detections += detections_this_frame
        
        # Annotate frame
        annotated_frame = results[0].plot()
        
        # Add frame info
        info_text = f'Frame: {frame_count} | Detections: {detections_this_frame}'
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save frame if output specified
        if output_path:
            out.write(annotated_frame)
        
        # Display frame
        cv2.imshow('Object Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections/frame_count:.2f}")

if __name__ == "__main__":
    # Example usage
    
    # Detect in single image
    # result_image = detect_objects_image('sample_image.jpg')
    # cv2.imshow('Detection Result', result_image)
    # cv2.waitKey(0)
    
    # Detect in video
    detect_objects_video('sample_video.mp4', 'output_detection.mp4')
```

## 🎯 Object Tracking Demo

```python
# examples/tracking_demo.py
"""
Advanced object tracking with DeepSORT
Maintains object IDs across frames
"""

import cv2
from ultralytics import YOLO
from deep_sort_realtime import DeepSort
import numpy as np
from collections import defaultdict

class ObjectTracker:
    def __init__(self, model_path='yolov5s.pt'):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(
            max_age=50,
            n_init=3,
            nn_budget=None,
            embedder_gpu=True,
            half=True
        )
        
        # Track statistics
        self.track_history = defaultdict(list)
        self.track_classes = {}
        self.active_tracks = set()
        
    def generate_color(self, track_id):
        """Generate consistent color for each track ID"""
        np.random.seed(int(track_id) % 1000)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def process_frame(self, frame):
        """Process single frame for tracking"""
        
        # Run detection
        results = self.model(frame)
        
        # Extract detections for DeepSORT
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Format for DeepSORT: [x, y, w, h]
                    detection = ([x1, y1, x2-x1, y2-y1], confidence, class_name)
                    detections.append(detection)
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Process tracks
        current_active = set()
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
                
            track_id = track.track_id
            current_active.add(track_id)
            
            # Get bounding box
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get center point
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Store track history
            self.track_history[track_id].append(center)
            
            # Keep only recent history (last 30 points)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id] = self.track_history[track_id][-30:]
            
            # Store class information
            if hasattr(track, 'det_class') and track.det_class:
                self.track_classes[track_id] = track.det_class
            
            # Generate color for this track
            color = self.generate_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and class
            class_name = self.track_classes.get(track_id, 'Unknown')
            label = f'ID: {track_id} | {class_name}'
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw track history (path)
            if len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], np.int32)
                cv2.polylines(frame, [points], False, color, 2)
        
        # Update active tracks
        self.active_tracks = current_active
        
        return frame
    
    def get_statistics(self):
        """Get tracking statistics"""
        return {
            'total_tracks': len(self.track_history),
            'active_tracks': len(self.active_tracks),
            'track_classes': dict(self.track_classes)
        }

def run_tracking_demo(video_path, output_path=None):
    """Run tracking demo on video"""
    
    tracker = ObjectTracker()
    cap = cv2.VideoCapture(video_path)
    
    # Video writer setup
    if output_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame
        tracked_frame = tracker.process_frame(frame)
        
        # Add statistics overlay
        stats = tracker.get_statistics()
        stats_text = [
            f"Frame: {frame_count}",
            f"Active Tracks: {stats['active_tracks']}",
            f"Total Tracks: {stats['total_tracks']}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(tracked_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        # Save frame
        if output_path:
            out.write(tracked_frame)
        
        # Display
        cv2.imshow('Object Tracking Demo', tracked_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    final_stats = tracker.get_statistics()
    print("\n=== Tracking Results ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique tracks: {final_stats['total_tracks']}")
    print(f"Final active tracks: {final_stats['active_tracks']}")
    print(f"Track classes: {final_stats['track_classes']}")

if __name__ == "__main__":
    # Run tracking demo
    run_tracking_demo('sample_video.mp4', 'output_tracking.mp4')
```

## 📍 Zone Analysis Demo

```python
# examples/zone_analysis_demo.py
"""
Zone-based analytics demo
Counts objects entering/exiting predefined zones
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime import DeepSort

class ZoneAnalyzer:
    def __init__(self, zones, model_path='yolov5s.pt'):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=30, n_init=3)
        
        self.zones = zones
        self.zone_counts = {i: {"entered": 0, "exited": 0} for i in range(len(zones))}
        self.object_paths = {}
        
    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def analyze_crossing(self, track_id, current_center):
        """Analyze zone crossings for a track"""
        if track_id not in self.object_paths:
            self.object_paths[track_id] = []
        
        self.object_paths[track_id].append(current_center)
        
        # Need at least 2 points to detect crossing
        if len(self.object_paths[track_id]) < 2:
            return
        
        prev_center = self.object_paths[track_id][-2]
        
        # Check each zone
        for zone_id, zone in enumerate(self.zones):
            prev_inside = self.point_in_polygon(prev_center, zone)
            curr_inside = self.point_in_polygon(current_center, zone)
            
            # Detect entry
            if not prev_inside and curr_inside:
                self.zone_counts[zone_id]["entered"] += 1
            
            # Detect exit
            elif prev_inside and not curr_inside:
                self.zone_counts[zone_id]["exited"] += 1
    
    def draw_zones(self, frame):
        """Draw zones on frame"""
        overlay = frame.copy()
        
        for i, zone in enumerate(self.zones):
            # Draw filled polygon with transparency
            color = self.get_zone_color(i)
            pts = np.array(zone, np.int32)
            cv2.fillPoly(overlay, [pts], color)
            
            # Draw border
            cv2.polylines(frame, [pts], True, (0, 0, 0), 2)
            
            # Add zone label
            center_x = int(np.mean([p[0] for p in zone]))
            center_y = int(np.mean([p[1] for p in zone]))
            cv2.putText(frame, f'Zone {i+1}', (center_x-30, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    def get_zone_color(self, zone_id):
        """Get consistent color for zone"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        return colors[zone_id % len(colors)]
    
    def draw_statistics(self, frame):
        """Draw zone statistics on frame"""
        # Create statistics panel
        panel_height = len(self.zones) * 60 + 40
        panel = np.zeros((panel_height, 300, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(panel, 'Zone Statistics', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = 50
        for zone_id, counts in self.zone_counts.items():
            color = self.get_zone_color(zone_id)
            
            # Zone info
            text = f'Zone {zone_id + 1}:'
            cv2.putText(panel, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Entered count
            entered_text = f'  In: {counts["entered"]}'
            cv2.putText(panel, entered_text, (10, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Exited count
            exited_text = f'  Out: {counts["exited"]}'
            cv2.putText(panel, exited_text, (10, y_offset + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            y_offset += 60
        
        # Overlay panel on frame
        frame[0:panel_height, 0:300] = panel
    
    def process_frame(self, frame):
        """Process frame for zone analysis"""
        # Run detection
        results = self.model(frame)
        
        # Extract detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detection = ([x1, y1, x2-x1, y2-y1], confidence, class_name)
                    detections.append(detection)
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Process tracks
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
                
            track_id = track.track_id
            bbox = track.to_ltrb()
            
            # Get center point
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            
            # Analyze zone crossings
            self.analyze_crossing(track_id, center)
            
            # Draw track
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f'ID: {track_id}', 
                       (int(bbox[0]), int(bbox[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cv2.circle(frame, center, 3, color, -1)
        
        # Draw zones and statistics
        self.draw_zones(frame)
        self.draw_statistics(frame)
        
        return frame

def run_zone_analysis(video_path, zones, output_path=None):
    """Run zone analysis on video"""
    
    analyzer = ZoneAnalyzer(zones)
    cap = cv2.VideoCapture(video_path)
    
    # Video writer setup
    if output_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame
        analyzed_frame = analyzer.process_frame(frame)
        
        # Save frame
        if output_path:
            out.write(analyzed_frame)
        
        # Display
        cv2.imshow('Zone Analysis Demo', analyzed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # Print results
    print("\n=== Zone Analysis Results ===")
    for zone_id, counts in analyzer.zone_counts.items():
        print(f"Zone {zone_id + 1}: {counts['entered']} entered, {counts['exited']} exited")

if __name__ == "__main__":
    # Define zones (example for traffic monitoring)
    zones = [
        [(100, 200), (300, 200), (300, 400), (100, 400)],  # Entry zone
        [(500, 200), (700, 200), (700, 400), (500, 400)]   # Exit zone
    ]
    
    # Run analysis
    run_zone_analysis('traffic_video.mp4', zones, 'zone_analysis_output.mp4')
```

## 🎨 Color Tracking Demo

```python
# examples/color_tracking_demo.py
"""
Color-based object tracking using HSV color space
Useful for tracking specific colored objects
"""

import cv2
import numpy as np
from collections import deque

class ColorTracker:
    def __init__(self, color_ranges):
        """
        Initialize color tracker
        color_ranges: dict with color names as keys and (lower, upper) HSV tuples as values
        """
        self.color_ranges = color_ranges
        self.object_paths = {color: deque(maxlen=50) for color in color_ranges}
        self.object_centers = {color: None for color in color_ranges}
        
    def create_mask(self, hsv_frame, color_name):
        """Create mask for specific color"""
        lower, upper = self.color_ranges[color_name]
        lower = np.array(lower)
        upper = np.array(upper)
        
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def find_largest_contour(self, mask):
        """Find the largest contour in mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                return largest_contour
        
        return None
    
    def get_color_for_name(self, color_name):
        """Get BGR color for visualization"""
        color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (255, 0, 255),
            'cyan': (255, 255, 0)
        }
        return color_map.get(color_name, (255, 255, 255))
    
    def process_frame(self, frame):
        """Process frame for color tracking"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Process each color
        for color_name in self.color_ranges:
            # Create mask
            mask = self.create_mask(hsv, color_name)
            
            # Find largest contour
            contour = self.find_largest_contour(mask)
            
            if contour is not None:
                # Get bounding box and center
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                
                # Update center and path
                self.object_centers[color_name] = center
                self.object_paths[color_name].append(center)
                
                # Get color for drawing
                draw_color = self.get_color_for_name(color_name)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                
                # Draw center point
                cv2.circle(frame, center, 5, draw_color, -1)
                
                # Draw label
                label = f'{color_name.capitalize()} Object'
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
                
                # Draw path
                if len(self.object_paths[color_name]) > 1:
                    points = np.array(list(self.object_paths[color_name]), np.int32)
                    cv2.polylines(frame, [points], False, draw_color, 2)
            else:
                # Object not found, clear current center
                self.object_centers[color_name] = None
        
        return frame
    
    def get_debug_masks(self, frame):
        """Get debug masks for each color"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks = {}
        
        for color_name in self.color_ranges:
            masks[color_name] = self.create_mask(hsv, color_name)
        
        return masks

def run_color_tracking_demo(video_path, output_path=None):
    """Run color tracking demo"""
    
    # Define color ranges in HSV
    color_ranges = {
        'red': ([0, 120, 70], [10, 255, 255]),      # Red objects
        'green': ([36, 50, 70], [89, 255, 255]),     # Green objects  
        'blue': ([100, 50, 70], [130, 255, 255]),    # Blue objects
        'yellow': ([20, 100, 100], [30, 255, 255])   # Yellow objects
    }
    
    tracker = ColorTracker(color_ranges)
    cap = cv2.VideoCapture(video_path)
    
    # Video writer setup
    if output_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame
        tracked_frame = tracker.process_frame(frame)
        
        # Add frame counter
        cv2.putText(tracked_frame, f'Frame: {frame_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save frame
        if output_path:
            out.write(tracked_frame)
        
        # Display
        cv2.imshow('Color Tracking Demo', tracked_frame)
        
        # Show debug masks (press 'd' to toggle)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            masks = tracker.get_debug_masks(frame)
            for color_name, mask in masks.items():
                cv2.imshow(f'{color_name} Mask', mask)
        elif key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nColor tracking completed! Processed {frame_count} frames.")

if __name__ == "__main__":
    # Run color tracking demo
    run_color_tracking_demo('colored_objects_video.mp4', 'color_tracking_output.mp4')
```

## 🎮 Interactive Controls

### Keyboard Shortcuts for Demos

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `p` | Pause/Resume |
| `s` | Save current frame |
| `d` | Toggle debug mode |
| `r` | Reset tracking |
| `+/-` | Adjust confidence threshold |
| `Space` | Step frame (when paused) |

### Mouse Controls (Zone Demo)

| Action | Control |
|--------|---------|
| Define zone vertex | Left click |
| Remove last vertex | Right click |
| Complete polygon | Enter key |
| Start analysis | 's' key |

## 📈 Performance Tips

1. **Resize frames** for faster processing:
   ```python
   frame = cv2.resize(frame, (640, 480))
   ```

2. **Skip frames** for real-time processing:
   ```python
   if frame_count % 2 == 0:  # Process every 2nd frame
       results = model(frame)
   ```

3. **Use smaller models** for edge deployment:
   ```python
   model = YOLO('yolov5n.pt')  # Nano version
   ```

4. **Batch processing** for multiple videos:
   ```python
   for video_file in video_list:
       process_video(video_file)
   ```

## 🔧 Customization Examples

### Custom Detection Classes

```python
# Filter for specific classes only
target_classes = ['person', 'car', 'bicycle']
filtered_detections = [det for det in detections 
                      if det[2] in target_classes]
```

### Custom Tracking Parameters

```python
# Aggressive tracking for crowded scenes
tracker = DeepSort(
    max_age=10,      # Shorter memory
    n_init=2,        # Faster confirmation
    nn_budget=50     # Limited feature storage
)

# Conservative tracking for sparse scenes  
tracker = DeepSort(
    max_age=100,     # Longer memory
    n_init=5,        # Slower confirmation
    nn_budget=None   # Unlimited features
)
```

### Custom Zone Shapes

```python
# Circular zone
def point_in_circle(point, center, radius):
    return np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2) <= radius

# Elliptical zone
def point_in_ellipse(point, center, a, b):
    x, y = point
    cx, cy = center
    return ((x - cx)/a)**2 + ((y - cy)/b)**2 <= 1
```

---

**🎯 Ready to experiment?** Try these examples and modify them for your specific use case!

For more advanced examples, check the [notebooks](../ColorSegmentation/) directory.
