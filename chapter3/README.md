# 📘 Chapter 3: Computer Vision & CNNs

Welcome to the third chapter of the AI Course! In this chapter, we transition from classical Machine Learning and simple Neural Networks to the exciting world of **Computer Vision**.

We will explore how machines "see" images, moving from the mathematical foundations of Convolution to building state-of-the-art systems for Object Detection and Segmentation.

---

## 🧠 Topics Covered

* **CNN Fundamentals:** Convolution, Stride, Padding, and Pooling.
* **Classic Architectures:** LeNet-5, AlexNet, and VGG-16.
* **Modern Architectures:** ResNet (Skip Connections), Inception, and EfficientNet.
* **Transfer Learning:** How to use pre-trained weights (ImageNet) for new tasks.
* **Object Detection:** Understanding YOLO (You Only Look Once) architecture.
* **Image Segmentation:** Semantic Segmentation with U-Net.

---

## 📂 Notebook Structure

This chapter is divided into educational notebooks (theory & code) and a capstone project.

### 1️⃣ `01_CNN_Foundations.ipynb`
**Focus:** Understanding the building blocks.
We manually implement Convolution filters (like Edge Detection) and visualize Feature Maps to understand what the network "sees."

### 2️⃣ `02_Classic_Architectures.ipynb`
**Focus:** The history of Deep Learning.
We implement **LeNet** from scratch and analyze **VGG** and the "Vanishing Gradient" problem that plagued early deep networks.

### 3️⃣ `03_Transfer_Learning_ResNet.ipynb`
**Focus:** Training deep models on small datasets.
We use **ResNet18** with pre-trained weights to classify images, a technique essential for real-world applications where data is scarce.

### 4️⃣ `04_Object_Detection_YOLO.ipynb`
**Focus:** Theory of Object Detection.
Introduction to Anchor Boxes, IoU (Intersection over Union), and running inference with **YOLOv8** to detect multiple objects in an image.

### 5️⃣ `05_Segmentation_UNet.ipynb`
**Focus:** Pixel-level precision.
Building the **U-Net** Encoder-Decoder architecture from scratch for semantic image segmentation.

---

## 🚀 Capstone Project: EcoVision

**The Intelligent Waste Sorting System**

In the `projects/EcoVision` folder, we apply all the concepts from this chapter to build a complete application using the **TACO Dataset**.

* **Phase 1:** Classifying trash (Bottle vs. Can) using **ResNet**.
* **Phase 2:** Detecting trash location using **YOLO**.
* **Phase 3:** Segmenting trash pixels using **U-Net**.

---

## 💻 Hardware Notes (Apple Silicon)

All code in this chapter is optimized for **Apple Silicon** chips.

We use the **MPS (Metal Performance Shaders)** backend to accelerate training on macOS. The notebooks include code to automatically detect your accelerator:

```python
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS backend)")
else:
    device = torch.device("cpu")

```

---

### 📚 Dependencies

To run these notebooks, install the required libraries:

```bash
pip install torch torchvision numpy matplotlib opencv-python ultralytics

```