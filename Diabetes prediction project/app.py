import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("gradient_boosting_classifier_model.pkl")

st.title("Diabetes Prediction App")
st.write("Enter the following details to predict diabetes:")

# Input fields for user
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prepare input for model (ONLY 8 features)
input_data = [
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age
]

input_df = pd.DataFrame([input_data], columns=[
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("Prediction: Diabetic")
    else:
        st.success("Prediction: Not Diabetic")