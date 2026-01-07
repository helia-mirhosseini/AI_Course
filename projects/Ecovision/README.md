
# 🌱 EcoVision: Intelligent Waste Sorting System

**EcoVision** is a comprehensive Deep Learning vision system designed to automate waste sorting for environmental sustainability. It leverages modern Computer Vision architectures to Classify, Detect, and Segment waste items in real-time.

This project is optimized for **Apple Silicon (M-series)** hardware, specifically utilizing Metal Performance Shaders (MPS) for GPU acceleration on the **MacBook Air M4**.

---

## 🚀 Project Overview

The system operates in three distinct phases, each solving a specific computer vision challenge:

1. **Phase 1: Classification (ResNet-18)**
* **Goal:** Identify the *type* of waste (e.g., Plastic vs. Paper vs. Metal).
* **Technique:** Transfer Learning using a pre-trained ResNet backbone.


2. **Phase 2: Detection (YOLOv8)**
* **Goal:** Locate litter in complex, messy scenes and draw bounding boxes.
* **Technique:** Real-time object detection using Ultralytics YOLOv8 (Nano/Small models).


3. **Phase 3: Segmentation (U-Net)**
* **Goal:** Generate pixel-perfect masks for waste items (e.g., exact shape of a crushed bottle).
* **Technique:** Semantic Segmentation using U-Net with a ResNet encoder.



---

## 🛠 Tech Stack & Hardware

* **Language:** Python 3.12
* **Frameworks:** PyTorch, Torchvision
* **Libraries:**
* `ultralytics` (YOLOv8)
* `segmentation-models-pytorch` (U-Net)
* `opencv-python`, `matplotlib`, `pandas`


* **Hardware:** Apple MacBook Air M4 (16GB RAM / 512GB SSD)
* **Acceleration:** Apple Metal Performance Shaders (MPS)

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/helia-mirhosseini/AI_Course.git
cd AI_Course/projects/EcoVision

```

### 2. Set Up Environment (Apple Silicon Optimized)

It is recommended to use a virtual environment to avoid conflicts.

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

Run the following commands to install the required libraries with MPS support:

```bash
# Core Torch libraries
pip install torch torchvision torchaudio

# Computer Vision specific libraries
pip install ultralytics segmentation-models-pytorch opencv-python matplotlib

```

---

## 🏃‍♀️ Usage

### Phase 1: Classification (ResNet)

To train the classifier on your dataset:

```python
python classifier.py

```

*Modify `classifier.py` to point to your specific data folders (`data/train`, `data/val`).*

### Phase 2: Detection (YOLOv8)

To train the object detector. Note the optimization flags used for M-series chips.

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
results = model.train(
    data='coco128.yaml',  # Replace with your custom dataset.yaml
    epochs=50, 
    imgsz=640, 
    device='mps',         # Enables M4 GPU
    batch=8,              # Optimized for 16GB RAM
    workers=0             # CRITICAL: Prevents data loader hanging on macOS
)

```

### Phase 3: Segmentation (U-Net)

To test the segmentation architecture:

```python
python unet_model.py

```

---

## 🍎 Apple M4 Optimization Notes

This project contains specific optimizations for training deep learning models on Apple Silicon (M4/M3/M2/M1) with 16GB RAM constraints:

1. **MPS Device:** All scripts explicitly set `device = torch.device("mps")` to utilize the Neural Engine/GPU instead of CPU.
2. **Worker Threads (`workers=0`):** PyTorch's `DataLoader` can cause bottlenecks on macOS when using multiple subprocesses. We set `workers=0` in YOLO training to force main-thread data loading, which significantly improves speed and stability on this hardware.
3. **Batch Size:** Training batch sizes are tuned (8-16) to fit within the Unified Memory architecture without triggering swap memory usage.

---

## 📊 Results

* **YOLOv8:** Successfully converged on test data (COCO8/128).
* **Inference Speed:** Achieving real-time performance (~8ms per image) on M4 Neural Engine.

---

## 📝 License

This project is for educational and research purposes.

```

### **How to add this to your project:**
1.  Go to your VS Code "EcoVision" folder.
2.  Create a new file named `README.md`.
3.  Paste the text above into it.
4.  Save it.
