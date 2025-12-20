
# Chapter 2 — Deep Learning Core Concepts

This chapter provides a **structured and concept-driven introduction to deep learning**, covering neural network foundations, computer vision, sequence modeling, and attention mechanisms.
The focus is on **understanding why things work**, not just how to run code.

---

## 🎯 Learning Objectives

By completing this chapter, you should be able to:

* Explain how artificial neurons form deep neural networks
* Choose appropriate activation functions and loss functions
* Understand and describe backpropagation using the chain rule
* Compare and use different optimizers (SGD, RMSprop, Adam)
* Diagnose overfitting and apply regularization techniques
* Understand CNNs and why they dominate computer vision
* Explain why vanilla RNNs fail and how LSTM/GRU fix them
* Understand attention mechanisms and self-attention
* Compare deep learning models with classic ML algorithms

---

## 📌 Chapter Structure

---

## Section 1 — Neural Network Foundations

This section builds the **core mathematical and conceptual foundation** of neural networks. Everything that follows (CNNs, RNNs, Transformers) relies on these ideas.

### Topics Covered

* **Perceptron & Artificial Neurons**

  * Weighted sum + bias
  * Nonlinearity as the key to expressive power

* **Activation Functions**

  * Sigmoid
  * Tanh
  * ReLU
  * Vanishing gradient intuition

* **Multi-Layer Perceptron (MLP)**

  * Fully connected networks
  * Universal approximation idea
  * Strengths and limitations

* **Loss Functions**

  * MSE (Regression)
  * Cross-Entropy (Classification)
  * Why the choice of loss matters

* **Backpropagation**

  * Chain rule across computation graphs
  * Forward pass vs backward pass
  * Autograd intuition (PyTorch)

* **Optimizers**

  * SGD
  * RMSprop
  * Adam
  * Speed vs generalization trade-offs

* **Regularization**

  * L1 & L2 (weight decay)
  * Dropout
  * Bias–variance trade-off

* **Batch Normalization**

  * Stabilizing training
  * Faster convergence
  * Train vs eval behavior

### Classic ML Comparison

| Deep Learning           | Classic ML                 |
| ----------------------- | -------------------------- |
| MLP                     | Logistic Regression        |
| L2 Regularization       | Ridge Regression           |
| L1 Regularization       | Lasso                      |
| Non-convex optimization | Convex optimization        |
| Feature learning        | Manual feature engineering |

---

## Section 2 — Computer Vision & CNNs

CNNs introduce **inductive bias** tailored to images: locality, translation invariance, and parameter sharing.

### Topics Covered

* **Convolution Fundamentals**

  * Kernels
  * Stride
  * Padding
  * Receptive fields

* **Pooling Layers**

  * Max Pooling
  * Average Pooling
  * Spatial downsampling

* **Classic CNN Architectures**

  * LeNet
  * AlexNet
  * VGG

* **Modern CNN Architectures**

  * ResNet (skip connections)
  * Inception (multi-scale features)
  * EfficientNet (compound scaling)

* **Transfer Learning**

  * Pretrained backbones
  * Freezing vs fine-tuning layers

* **Object Detection**

  * YOLO
  * SSD
  * Faster R-CNN

* **Image Segmentation**

  * U-Net
  * Mask R-CNN

### Why CNNs Beat MLPs on Images

* Exploit spatial locality
* Share parameters
* Fewer parameters than MLPs
* Better generalization on images

---

## Section 3 — Sequence Modeling & RNNs

This section focuses on **ordered data**, such as time series and text.

### Topics Covered

* **Vanilla RNN**

  * Hidden state propagation
  * Short-term memory

* **Gradient Problems**

  * Vanishing gradients
  * Exploding gradients

* **LSTM**

  * Forget, input, and output gates
  * Long-term memory retention

* **GRU**

  * Simplified gated architecture
  * Faster training

* **Bidirectional RNNs**

  * Forward + backward context

* **Time-Series Forecasting**

  * Rolling windows
  * Leakage prevention
  * Evaluation pitfalls

### Classic ML Comparison

| Deep Learning      | Classic Time Series |
| ------------------ | ------------------- |
| LSTM / GRU         | ARIMA / ETS         |
| Nonlinear patterns | Linear assumptions  |
| Data-hungry        | Data-efficient      |
| Flexible           | Interpretable       |

---

## Section 4 — NLP Basics & Attention

This section explains **why attention changed everything** in NLP.

### Topics Covered

* **Text Preprocessing**

  * Tokenization
  * Lemmatization

* **Word Embeddings**

  * Word2Vec
  * GloVe
  * Semantic similarity

* **Sequence-to-Sequence Models**

  * Encoder–Decoder structure
  * Bottleneck problem

* **Attention Mechanisms**

  * Bahdanau Attention
  * Luong Attention
  * Dynamic context vectors

* **Self-Attention**

  * Token-to-token interactions
  * Parallel computation
  * Foundation of Transformers

---

## 🧠 Practical Study Strategy

Recommended learning order:

1. **Neural Network Foundations** (must be solid)
2. **CNNs** (clear performance gains)
3. **RNNs & LSTM/GRU** (understand limitations)
4. **Attention & Self-Attention** (modern NLP core)

Best practice:

* Change **one parameter at a time**
* Compare results against **classic ML baselines**
* Track **validation performance**, not just training loss

---

## 🔬 Suggested Mini-Projects

* **MLP** on Fashion-MNIST and California Housing
  (activation, optimizer, dropout ablation)

* **Transfer Learning** with ResNet on a small image dataset

* **Time-Series Forecasting** using LSTM with rolling evaluation

* **Seq2Seq with Attention** on a toy text task

---

## 🍎 Notes for Apple Silicon (MPS)

* Prefer `mps` device when available
* Ensure model and tensors are on the same device
* Minor numerical differences vs CUDA are expected

---

## ✅ End-of-Chapter Checklist

You should confidently answer:

* Why are activation functions necessary?
* Why is cross-entropy preferred for classification?
* What exactly does backpropagation compute?
* When should you prefer SGD over Adam?
* How do dropout and weight decay prevent overfitting?
* Why do CNNs dominate vision tasks?
* Why do vanilla RNNs fail on long sequences?
* What problem does attention solve?
* Why is self-attention so powerful?

