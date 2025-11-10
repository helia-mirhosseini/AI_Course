
# Cancer Classification Project — ML Pipeline

This repository demonstrates a **complete machine learning workflow** for cancer diagnosis prediction using structured medical data.
It includes **data preprocessing, feature engineering, model training, evaluation, and pipeline automation** for reproducibility and scalability.

---

## 📂 Project Structure

```
cancer_classification/
│
├── cancer_new.csv              # Clean Dataset used for model training and evaluation
├── classification.ipynb        # Notebook for model development & comparison
├── pipeline.ipynb              # End-to-end ML pipeline automation
└──README.md                    # Project documentation
 
```

---

## 🧬 Project Overview

The goal of this project is to **predict whether a tumor is malignant or benign** using classical supervised learning algorithms.
The workflow covers the full ML lifecycle — from **data preprocessing** to **model evaluation and pipeline creation**.

### Key Objectives:

1. Explore and clean the dataset (`cancer_new.csv`)
2. Apply feature scaling and encoding
3. Train and compare classification models
4. Automate the process with a reusable **pipeline**
5. Evaluate performance metrics and visualize results

---

## ⚙️ Implementation Details

### 1. **Data Preprocessing**

* Handled missing values and outliers
* Encoded categorical features
* Standardized numerical features for algorithms sensitive to scale
* Split dataset into training and testing subsets

### 2. **Model Training**

`classification.ipynb` includes multiple supervised algorithms:

* Logistic Regression
* Random Forest Classifier
* Support Vector Machine (SVM)
* Gradient Boosting / XGBoost (if installed)

Each model is trained, tuned, and evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### 3. **Pipeline Construction**

`pipeline.ipynb` builds a reproducible **Scikit-learn pipeline** that includes:

* Preprocessing steps (scaling, encoding)
* Model training
* Cross-validation
* Automatic performance reporting

This ensures that data transformations and model training are consistent for any new dataset.

### 4. **Evaluation & Visualization**

The notebooks include:

* Confusion matrices
* ROC curves
* Feature importance plots
* Cross-validation scores

---

## 📊 Example Results

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.96     | 0.95      | 0.97   | 0.96     |
| Random Forest       | 0.98     | 0.98      | 0.99   | 0.98     |
| SVM (RBF kernel)    | 0.97     | 0.96      | 0.98   | 0.97     |

*(Values illustrative; see notebook outputs for actual metrics.)*

---

## 🧱 Dependencies

* Python ≥ 3.10
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn
* joblib (for saving models)
* xgboost (optional)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/cancer_classification.git
   cd cancer_classification
   ```

2. **Open and run the notebooks**

   ```bash
   jupyter notebook classification.ipynb
   jupyter notebook pipeline.ipynb
   ```

3. **Inspect results**

   * Metrics and charts appear inline.
   * Saved models can be found under `models/` if joblib saving is enabled.

---

## 🌐 Next Steps — Web Application

The next goal is to extend this project into a **web application** where users can upload a CSV file or manually input patient data to get an instant prediction.

### Planned Features:

* **Backend:** FastAPI or Flask endpoint that loads the trained pipeline and exposes a `/predict` route.
* **Frontend:** Streamlit or React dashboard to input patient data and display:

  * Predicted cancer type (malignant/benign)
  * Probability/confidence score
  * Feature importance chart
* **Deployment:** Docker container or cloud deployment (e.g., Render, Railway, or AWS EC2)

This step transforms the ML pipeline into an interactive diagnostic web tool for real-time predictions.

---

## 🧩 Author

**Helia Mirhosseini**
Machine Learning Engineer
*Focused on deploying interpretable ML systems for scientific and medical applications.*
