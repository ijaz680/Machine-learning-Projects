# 🧠 Diabetes Prediction Machine Learning Project

This project is a complete **end-to-end Machine Learning pipeline** for predicting whether a patient has diabetes based on medical attributes.

It includes:
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Outlier Detection
- Model Training
- Model Evaluation
- Model Comparison
- Model Saving

---

## 📌 Problem Statement

Predict whether a patient is diabetic or not based on health-related features such as:
- Glucose level
- Blood pressure
- BMI
- Insulin level
- Age
- Pregnancies

This is a **binary classification problem**:
- `0` → Non-Diabetic
- `1` → Diabetic

---

## 📊 Dataset

- Dataset: `diabetes.csv`
- Total Records: 768
- Features: 8 input features + 1 target column (`Outcome`)

---

## ⚙️ Technologies Used

- Python 🐍
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Statsmodels
- Joblib

---

## 🔍 Project Workflow

### 1. Data Loading
- Loaded dataset using pandas
- Checked structure and summary statistics

### 2. Exploratory Data Analysis (EDA)
- Distribution plots
- Correlation heatmap
- Outcome class distribution
- Group analysis (mean, max values)

### 3. Data Preprocessing
- Handled missing values using median imputation
- Replaced invalid zero values with NaN
- Filled missing values based on outcome groups

### 4. Feature Engineering
- Created BMI categories:
  - Underweight
  - Normal
  - Overweight
  - Obesity levels
- Created glucose level categories
- Created insulin score feature
- Applied One-Hot Encoding

### 5. Outlier Detection
- Used:
  - IQR method
  - Local Outlier Factor (LOF)
- Removed extreme outliers

### 6. Feature Scaling
- Applied:
  - RobustScaler
  - StandardScaler

---

## 🤖 Machine Learning Models Used

The following models were trained and tested:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest (optional available in imports)
- Gradient Boosting Classifier
- XGBoost Classifier

---

## 📈 Model Evaluation Metrics

Models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-score
- Classification Report

---

## 🏆 Best Model Performance

| Model | Accuracy |
|------|--------|
| Gradient Boosting Classifier | ⭐ Highest (~90%) |
| KNN | ~88% |
| XGBoost | ~87% |
| SVM | ~86% |
| Decision Tree | ~85% |
| Logistic Regression | ~81% |

---

## 💾 Model Saving

Final trained model saved using Joblib:

```python
joblib.dump(gbc, "gradient_boosting_classifier_model.pkl")
