# GDP Prediction Project 🌍

## Project Overview
This project is a Machine Learning based GDP Prediction System developed using Python. The system analyzes country data such as population, literacy rate, agriculture, industry, birthrate, and other economic features to predict the GDP per capita of countries.

The project uses different Machine Learning models including:

- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor

Dataset used:
`countries of the world.csv`

---

# Features 🚀

- Data Cleaning and Preprocessing
- Missing Value Handling
- Data Visualization
- Correlation Analysis
- GDP Prediction using ML Models
- Model Evaluation
- Save Model using Pickle

---

# Technologies Used 💻

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle

---

# Dataset Information 📊

The dataset contains:

- Country Name
- Region
- Population
- Area
- Literacy Rate
- Birthrate
- Deathrate
- Agriculture
- Industry
- Service
- GDP per Capita

---

# Project Workflow 🔄

## 1. Load Dataset
Dataset is loaded using Pandas.

## 2. Data Cleaning
- Converted object columns to float
- Removed commas from values
- Filled missing values

## 3. Data Analysis
Created graphs and charts for:
- GDP comparison
- Literacy analysis
- Region analysis
- Correlation analysis

## 4. Model Training
Trained three models:

### Linear Regression
Basic prediction model.

### Random Forest Regressor
Best performing model.

### Decision Tree Regressor
Good training accuracy but overfitting issue.

---

# Model Performance 📈

| Model | R² Score | RMSE |
|-------|-----------|------|
| Linear Regression | 0.74 | 4892 |
| Random Forest | 0.77 | 4817 |
| Decision Tree | 0.59 | 6572 |

Best Model:
✅ Random Forest Regressor

---

# Save Model 💾

```python
import pickle
pickle.dump(model_dtr, open('model_dtr.pkl', 'wb'))
```

---

# Installation ⚙️

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# Run Project ▶️

Run Jupyter Notebook:

```bash
jupyter notebook
```

---

# Future Improvements 🔥

- Add Streamlit Web App
- Improve Model Accuracy
- Deploy Online
- Add More Features

---

# Author 👨‍💻

Developed by:
Ijaz Ur Rahman

---

# Conclusion 📌

This project shows how Machine Learning can predict GDP per capita using country economic data. Among all models, Random Forest Regressor gave the best performance.
