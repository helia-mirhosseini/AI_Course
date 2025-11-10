# Machine Learning Algorithms – Persian Explanatory Notebooks

This repository contains a collection of Jupyter notebooks demonstrating key **Machine Learning algorithms** with **step-by-step Persian explanations**.
Each notebook focuses on one algorithm, combining theoretical background, implementation in Python, visualization, and interpretation of results.

---

## 📘 About the Project

These notebooks are designed as both an educational and reference resource for students and enthusiasts of **Machine Learning**, **Data Science**, and **Artificial Intelligence**.
All explanations are written in **Persian (Farsi)** to make complex mathematical and algorithmic concepts more accessible to Persian-speaking learners.

---

## 🧠 Algorithms Covered

The collection includes (but is not limited to):

| Category                      | Algorithms                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------ |
| **Supervised Learning**       | Linear Regression, Logistic Regression, Decision Tree, Random Forest, KNN, SVM |
| **Unsupervised Learning**     | K-Means, DBSCAN, PCA, LDA                                                      |
| **Neural Networks**           | Perceptron, Adaline, Multi-Layer Perceptron (MLP)                              |
| **Optimization & Evaluation** | Gradient Descent, Confusion Matrix, ROC Curve, Cross-Validation                |

Each notebook typically contains:

1. **Theoretical overview** (in Persian)
2. **Mathematical formulation**
3. **Python implementation** using `NumPy`, `pandas`, `scikit-learn`, `matplotlib`
4. **Visualization** and **result interpretation**

---

## 🗂 Folder Structure

```
AI_Course/
│
├── README.md
├── chapter1/
│   ├── Decision Tree.ipynb
│   ├── KNN.ipynb
│   ├── Logistic_Regression.ipynb
│   ├── PCA_LDA.ipynb
│   ├── Random Forest.ipynb
│   └── SVM.ipynb
└── projects/
    ├── California_Housing/
    └── Breast_Cancer/
```

---

## 🧩 Projects

In addition to the algorithm notebooks, this repository now includes **two applied Machine Learning projects** that demonstrate how the studied algorithms can be used in real-world scenarios.

### 1. **California Housing Price Prediction**

A complete **end-to-end regression pipeline** built on the **California Housing dataset**, predicting median house prices using demographic and geographical data.
Key highlights:

* Data preprocessing and feature engineering
* Model comparison (Linear Regression, Random Forest, XGBoost)
* Deployment-ready pipeline (`best_model.joblib`)
* Interactive web interface built with **Flask, HTML, and CSS**
* Visual map of California included for spatial interpretation

This project shows how core regression models can be scaled into a **functional web application**.

---

### 2. **Breast Cancer Classification**

A full **classification workflow** applied to the **Breast Cancer dataset**, predicting whether a tumor is benign or malignant.
Key highlights:

* Exploratory data analysis and visualization
* Feature scaling and encoding
* Model training with Logistic Regression, SVM, Random Forest
* Performance metrics: accuracy, precision, recall, F1-score, ROC-AUC
* Pipeline creation for automated preprocessing and inference

This project emphasizes **medical data modeling**, focusing on interpretability and accuracy in sensitive decision-making contexts.

---

## 🎯 Purpose

Together, these projects demonstrate how theoretical ML algorithms evolve into **real-world predictive systems**, reinforcing both conceptual understanding and practical implementation skills.

---

## 🧱 Dependencies

* Python ≥ 3.10
* numpy, pandas, scikit-learn, matplotlib, seaborn
* joblib, flask (for web app)
* xgboost (optional, for experiments)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

**Helia Mirhosseini**
Machine Learning Engineer | AI Researcher
Creating bilingual educational content and real-world ML applications bridging **theory and practice**.
