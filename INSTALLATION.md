# 🔧 Installation Guide

This guide provides detailed installation instructions for the Video Analytics framework.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 5GB free space for models and datasets
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

### Software Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU acceleration)
- **Git**: For repository cloning

## Step-by-Step Installation

### 1. Python Environment Setup

```bash
# Check Python version
python --version

# Create virtual environment (recommended)
python -m venv video_analytics_env

# Activate virtual environment
# Windows:
video_analytics_env\Scripts\activate
# Linux/macOS:
source video_analytics_env/bin/activate
```

### 2. Clone Repository

```bash
git clone https://github.com/your-username/Video-Analytics.git
cd Video-Analytics
```

### 3. Install Core Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install OpenCV
pip install opencv-python opencv-contrib-python

# Install Ultralytics YOLO
pip install ultralytics

# Install DeepSORT
pip install deep-sort-realtime

# Install other dependencies
pip install numpy matplotlib pandas seaborn jupyter
```

### 4. Download Pre-trained Models

```bash
# Create models directory
mkdir models

# Download YOLOv5 models
wget -P models/ https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
wget -P models/ https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
wget -P models/ https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
wget -P models/ https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt

# Download YOLOv9 models
wget -P models/ https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
wget -P models/ https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
```

### 5. Verify Installation

```python
# test_installation.py
import torch
import cv2
import numpy as np
from ultralytics import YOLO

print("🔍 Checking installation...")

# Check Python version
print(f"✅ Python: {sys.version}")

# Check PyTorch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")

# Check OpenCV
print(f"✅ OpenCV: {cv2.__version__}")

# Test YOLO model loading
try:
    model = YOLO('models/yolov5s.pt')
    print("✅ YOLOv5 model loaded successfully")
except Exception as e:
    print(f"❌ YOLOv5 loading failed: {e}")

print("🎉 Installation verification complete!")
```

```bash
python test_installation.py
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA/GPU Issues

**Problem**: CUDA not detected or GPU not being used

**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. OpenCV Import Error

**Problem**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution** (Linux):
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

#### 3. Memory Issues

**Problem**: Out of memory errors during model loading

**Solution**:
```python
# Use smaller models
model = YOLO('yolov5n.pt')  # Instead of yolov5l.pt

# Reduce batch size
results = model(image, batch_size=1)

# Use CPU if GPU memory is insufficient
device = 'cpu'
```

#### 4. Module Not Found Errors

**Problem**: `ModuleNotFoundError` for various packages

**Solution**:
```bash
# Install missing packages
pip install deep-sort-realtime
pip install filterpy  # If needed for tracking
pip install scipy     # If needed for calculations
```

## Platform-Specific Instructions

### Windows 10/11

1. Install Visual Studio Build Tools (for some packages)
2. Ensure Python is added to PATH
3. Use PowerShell or Command Prompt
4. May need to install Microsoft Visual C++ Redistributable

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv
sudo apt-get install libgl1-mesa-glx libglib2.0-0
sudo apt-get install ffmpeg  # For video processing
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Install system dependencies
brew install ffmpeg
```

## Docker Installation (Alternative)

If you prefer using Docker:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Download models
RUN mkdir models && \
    wget -P models/ https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

EXPOSE 8000

CMD ["python", "app.py"]
```

```bash
# Build and run Docker container
docker build -t video-analytics .
docker run -p 8000:8000 video-analytics
```

## Performance Optimization

### GPU Optimization

```python
# Enable mixed precision for faster inference
import torch
torch.backends.cudnn.benchmark = True

# Use TensorRT for NVIDIA GPUs (advanced)
model.export(format='engine')  # Creates TensorRT engine
```

### CPU Optimization

```python
# Enable OpenMP for CPU parallelization
import os
os.environ['OMP_NUM_THREADS'] = '4'

# Use optimized OpenCV build
# Install opencv-python-headless for server environments
pip uninstall opencv-python
pip install opencv-python-headless
```

## Development Setup

For development and contribution:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Code formatting
black .
isort .
```

## Next Steps

After successful installation:

1. 📖 Read the main [README.md](README.md) for usage instructions
2. 🚀 Try the [Quick Start Guide](QUICKSTART.md)
3. 📊 Explore the [Examples](examples/) directory
4. 🔬 Run the [Jupyter notebooks](notebooks/) for interactive demos

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Search [GitHub Issues](https://github.com/your-username/Video-Analytics/issues)
3. Create a new issue with detailed error information
4. Join our [Discord community](https://discord.gg/your-server) for help

---

**Need help?** Contact us at your.email@university.edu
