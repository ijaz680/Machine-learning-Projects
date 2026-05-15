# 🧱 Concrete Strength Prediction - Machine Learning Project

## 📌 Project Overview
This project predicts the **compressive strength of concrete** using machine learning models.  
It uses different regression algorithms and compares their performance to select the best model.

---

## 🎯 Objective
To build a regression model that predicts concrete strength based on its chemical and physical components.

---

## 📊 Dataset
- Dataset: Concrete Data
- File: `concrete_data.csv`
- Total Records: 1030 rows
- Features: 8 input variables
- Target: Strength

---

## 🧪 Input Features
- Cement
- Blast Furnace Slag
- Fly Ash
- Water
- Superplasticizer
- Coarse Aggregate
- Fine Aggregate
- Age

---

## 🛠️ Tools & Libraries Used
- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- XGBoost
- Joblib

---

## 🔍 Exploratory Data Analysis (EDA)
- Checked dataset shape, null values, duplicates
- Statistical summary using `describe()`
- Correlation heatmap to understand feature relationships
- Distribution plots for feature analysis

---

## ⚙️ Data Preprocessing
- No missing values in dataset
- Detected 25 duplicate rows
- Applied:
  - PowerTransformer (for normal distribution)
  - StandardScaler (for feature scaling)

---

## 📊 Data Visualization
- Distribution plots for all features
- Q-Q plots to check normality
- Heatmap for correlation analysis

---

## 🤖 Models Used
The following regression models were trained and compared:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Decision Tree Regressor
- XGBoost Regressor

---

## 📈 Model Performance

| Model              | R² Score | MSE (Approx) |
|-------------------|----------|--------------|
| Linear Regression | 0.81     | 47.57        |
| Lasso Regression  | 0.79     | 53.25        |
| Ridge Regression  | 0.81     | 47.57        |
| Decision Tree     | 0.75     | 63.50        |
| XGBoost           | 0.79     | 52.82        |

---

## 🏆 Best Model
- **Linear Regression**
- R² Score: **0.815**

---

## 💾 Model Saving
```python
import joblib
joblib.dump(lr, "concrete_prediction")
