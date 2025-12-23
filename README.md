# Machine Learning Algorithms – Persian Explanatory Notebooks

This repository contains a collection of Jupyter notebooks demonstrating key **Machine Learning algorithms** with **step-by-step Persian explanations**.
Each notebook focuses on one algorithm, combining theoretical background, implementation in Python, visualization, and interpretation of results.

---

## 📘 About the Project

These notebooks are designed as both an educational and reference resource for students and enthusiasts of **Machine Learning**, **Data Science**, and **Artificial Intelligence**.
All explanations are written in **Persian (Farsi)** to make complex mathematical and algorithmic concepts more accessible to Persian-speaking learners.

---

## 🧠 Curriculum & Chapters

The course is structured into four main chapters, progressing from classical algorithms to advanced Deep Learning:

| Chapter | Topic | Key Concepts |
| :--- | :--- | :--- |
| **1. Classical ML** | Supervised & Unsupervised | Linear Regression, Decision Trees, KNN, SVM, K-Means, PCA |
| **2. Neural Networks** | Foundations of Deep Learning | Perceptron, Adaline, Multi-Layer Perceptron (MLP), Backpropagation |
| **3. Computer Vision** | CNNs & Visual Recognition | Convolutions, Pooling, ResNet, YOLO (Object Detection), U-Net (Segmentation) |
| **4. Sequence Models** | RNNs & Time-Series | Vanilla RNN, LSTM, GRU, Bidirectional RNNs, Time-Series Forecasting |

Each notebook typically contains:

1. **Theoretical overview** (in Persian)
2. **Mathematical formulation**
3. **Python implementation** using `PyTorch`, `scikit-learn`, `NumPy`, `pandas`
4. **Visualization** and **result interpretation**

---

## 🗂 Folder Structure

```text
AI_Course/
│
├── README.md
│
├── chapter1/                     # Classical Machine Learning
│   ├── Decision Tree.ipynb
│   ├── KNN.ipynb
│   ├── SVM.ipynb
│   └── ...
│
├── chapter2/                     # Neural Network Foundations
│   ├── Perceptron.ipynb
│   ├── Adaline.ipynb
│   └── MLP_Backpropagation.ipynb
│
├── chapter3_ComputerVision/      # Deep Learning for Vision (CNNs)
│   ├── 01_CNN_Foundations.ipynb
│   ├── 02_Classic_Architectures.ipynb
│   ├── 04_Object_Detection_Theory.ipynb
│   └── ...
│
├── chapter4_SequenceModeling/    # Deep Learning for Sequences (RNNs)
│   ├── 01_Time_Series_Basics.ipynb
│   ├── 02_LSTM_vs_GRU.ipynb
│   └── ...
│
└── projects/                     # Applied End-to-End Projects
    ├── California_Housing/       (Regression Pipeline)
    ├── Breast_Cancer/            (Medical Classification)
    ├── EcoVision/                (Computer Vision - Waste Sorting)
    └── EnergyPulse/              (Time-Series Forecasting)

```

---

## 🧩 Projects

This repository applies the theoretical concepts in **four major real-world projects**:

### 1. **California Housing Price Prediction** (Classical ML)

A complete **regression pipeline** predicting house prices.

* **Tech:** Scikit-Learn, Flask.
* **Key Skills:** Feature Engineering, Model Deployment, Web App integration.

### 2. **Breast Cancer Classification** (Classical ML)

A critical **medical diagnostic workflow** for tumor classification.

* **Tech:** SVM, Random Forest.
* **Key Skills:** Sensitivity/Specificity analysis, ROC-AUC, Handling Imbalanced Data.

### 3. **EcoVision: Intelligent Waste Sorting** (Computer Vision)

A comprehensive **Deep Learning vision system** for environmental sustainability.

* **Tech:** PyTorch, YOLOv8, U-Net, ResNet.
* **Phases:**
1. **Classification:** Identifying waste types (Transfer Learning).
2. **Detection:** Locating litter in images (YOLO).
3. **Segmentation:** Pixel-perfect waste masking (U-Net).



### 4. **EnergyPulse: Grid Load Forecasting** (Sequence Modeling)

A **Time-Series Forecasting** system for energy consumption.

* **Tech:** PyTorch, LSTM, GRU.
* **Key Skills:** Sequence windowing, Handling vanishing gradients, Long-term dependency modeling.

---

## 🎯 Purpose

Together, these projects demonstrate how theoretical ML algorithms evolve into **real-world predictive systems**, reinforcing both conceptual understanding and practical implementation skills.

---

## 🧱 Dependencies

* Python ≥ 3.10
* **Core:** numpy, pandas, matplotlib, seaborn
* **ML:** scikit-learn, xgboost
* **Deep Learning:** torch, torchvision, ultralytics (YOLO)
* **App:** joblib, flask

Install dependencies:

```bash
pip install -r requirements.txt

```

---

## ✍️ Author

**Helia Mirhosseini**
Machine Learning Engineer
Creating bilingual educational content and real-world ML applications bridging **theory and practice**.
