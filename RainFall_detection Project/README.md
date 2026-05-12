# 🌧️ Rainfall Prediction using Machine Learning

## 📌 Project Overview
This project predicts whether it will **rain or not** using weather data.  
It is built using **Machine Learning (Random Forest Classifier)** with hyperparameter tuning and data balancing techniques.

---

## 🎯 Objective
To classify weather conditions and predict:
- 🌧️ Rainfall (Yes / No)

based on meteorological features.

---

## 📊 Dataset
- Dataset: Rainfall Dataset (`Rainfall.csv`)
- Total Records: 366 rows
- Target Column: `rainfall`

---

## 🌦️ Features Used
- Pressure
- Dewpoint
- Humidity
- Cloud cover
- Sunshine
- Wind direction
- Wind speed

---

## 🛠️ Tools & Libraries Used
- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Pickle

---

## 🔄 Data Preprocessing

### Data Cleaning
- Removed unnecessary column (`day`)
- Fixed column name spaces
- Handled missing values:
  - winddirection → mode
  - windspeed → median

### Encoding
- Converted target variable:
  - Yes → 1
  - No → 0

---

## 📊 Exploratory Data Analysis (EDA)
- Histograms for feature distribution
- Boxplots for outlier detection
- Correlation heatmap
- Class distribution visualization

---

## ⚖️ Handling Imbalanced Data
- Applied **Downsampling**
- Balanced classes:
  - Rain = 117
  - No Rain = 117

---

## 🤖 Model Training
- Algorithm used: **Random Forest Classifier**
- Hyperparameter tuning using **GridSearchCV**
- Cross-validation applied (5-fold CV)

### Best Parameters:
- n_estimators: 100
- max_depth: 5
- min_samples_leaf: 2
- bootstrap: True

---

## 📈 Model Performance

### Cross Validation Score:
- ~0.81 (81%)

### Test Accuracy:
- ~0.72 (72%)

### Confusion Matrix:
- Model performs balanced classification for both classes

---

## 🔍 Classification Report
- Precision, Recall, and F1-score evaluated for both classes:
  - Rain (1)
  - No Rain (0)

---

## 🔮 Prediction Example
Example input:
```python
(1015, 9, 19.9, 95, 81, 0.0, 40.0, 13.7)
