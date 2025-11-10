import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("my_model_titanic")

st.set_page_config(
    page_title="🚢 Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #0077b6; color: white;}
    .st-bb {background-color: #caf0f8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival chance.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        Pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        Sex = st.selectbox("Sex", ["Male", "Female"])
        Age = st.slider("Age", 0, 80, 25)
        SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
    with col2:
        Parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
        Fare = st.number_input("Fare Paid", 0.0, 600.0, 32.0)
        Embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

    submitted = st.form_submit_button("Predict Survival")

if submitted:
    # Encode categorical variables as in your training
    sex_encoded = 0 if Sex == "Male" else 1
    embarked_encoded = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[Embarked]

    # Arrange features in the same order as training
    features = np.array([[Pclass, sex_encoded, Age, SibSp, Parch, Fare, embarked_encoded]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.success(f"🎉 The passenger is **likely to survive**! (Probability: {proba:.2%})")
    else:
        st.error(f"😢 The passenger is **unlikely to survive**. (Probability: {proba:.2%})")

    st.markdown(
        """
        <hr>
        <small>
        <b>Note:</b> This prediction is based on a logistic regression model trained on Titanic data.<br>
        For educational purposes only.
        </small>
        """,
        unsafe_allow_html=True,
    )