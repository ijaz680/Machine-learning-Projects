# 🚢 Titanic Survival Prediction - Machine Learning Project

## 📌 Project Overview
This project predicts whether a passenger survived the Titanic disaster using Machine Learning (Logistic Regression).  
It includes data cleaning, visualization, feature engineering, model training, and evaluation.

---

## 🧠 Objective
To build a model that predicts passenger survival based on:
- Passenger Class (Pclass)
- Gender (Sex)
- Age
- Fare
- Embarked location
- Family features (SibSp, Parch)

---

## 📊 Dataset
- Dataset: Titanic Dataset
- Source: Kaggle
- File used: `train.csv`

---

## 🛠️ Tools & Libraries Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 🔄 Project Workflow

### 1. Data Loading
- Loaded dataset using pandas

### 2. Data Cleaning
- Removed `Cabin` column (many missing values)
- Filled missing values:
  - Age → Mean value
  - Embarked → Mode value

### 3. Data Visualization
- Survival count
- Gender-wise survival
- Passenger class survival

### 4. Data Preprocessing
- Converted categorical values:
  - Sex: male = 0, female = 1
  - Embarked: S = 0, C = 1, Q = 2
- Dropped unnecessary columns:
  - Name, Ticket, PassengerId

---

## 🤖 Model Training
- Algorithm: Logistic Regression
- Train/Test Split: 80% / 20%

---

## 📈 Model Evaluation

### Training Accuracy
- ~80.7%

### Testing Accuracy
- ~78.2%

---

## 💾 Model Saving
```python
import joblib
joblib.dump(model, "my_model_titanic")
