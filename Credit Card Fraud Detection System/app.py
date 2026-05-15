import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("my_model.pkl")

st.title("Credit Card Fraud Detection")

st.write("Enter transaction details:")

# Add Time input
Time = st.number_input("Time")

V1 = st.number_input("Feature 1")
V2 = st.number_input("Feature 2")
V3 = st.number_input("Feature 3")
V4 = st.number_input("Feature 4")
V5 = st.number_input("Feature 5")
V6 = st.number_input("Feature 6")
V7 = st.number_input("Feature 7")
V8 = st.number_input("Feature 8")
V9 = st.number_input("Feature 9")
V10 = st.number_input("Feature 10")
V11 = st.number_input("Feature 11")
V12 = st.number_input("Feature 12")
V13 = st.number_input("Feature 13")
V14 = st.number_input("Feature 14")
V15 = st.number_input("Feature 15")
V16 = st.number_input("Feature 16")
V17 = st.number_input("Feature 17")
V18 = st.number_input("Feature 18")
V19 = st.number_input("Feature 19")
V20 = st.number_input("Feature 20")
V21 = st.number_input("Feature 21")
V22 = st.number_input("Feature 22")
V23 = st.number_input("Feature 23")
V24 = st.number_input("Feature 24")
V25 = st.number_input("Feature 25")
V26 = st.number_input("Feature 26")
V27 = st.number_input("Feature 27")
V28 = st.number_input("Feature 28")
Amount = st.number_input("Amount")

if st.button("Predict"):
    features = np.array([[Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount]])
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction.")