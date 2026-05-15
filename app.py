import streamlit as st
import numpy as np
import joblib
import pandas as pd

# -----------------------------
# 🎨 PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Healthcare Disease Predictor",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# 🎨 CUSTOM CSS (ATTRACTIVE UI)
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #00ffcc;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cfcfcf;
    }
    .box {
        padding: 20px;
        border-radius: 15px;
        background-color: #1c1f26;
        box-shadow: 0px 0px 10px rgba(0,255,204,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🏥 HEADER
# -----------------------------
st.markdown('<div class="title">🏥 Healthcare Disease Pattern Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system to predict disease risk based on patient health data</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------
# 📦 LOAD MODEL & SAFE-SCALER
# -----------------------------
model = joblib.load("best_healthcare_model.pkl")

# Try loading external scaler; if missing, try to extract from pipeline/model
scaler = None
try:
    scaler = joblib.load("scaler.pkl")
except Exception:
    # look for scaler inside pipeline steps (common names)
    try:
        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                n = name.lower()
                cls = step.__class__.__name__.lower()
                if "scaler" in n or cls.startswith(("standardscaler", "minmaxscaler", "robustscaler")):
                    scaler = step
                    break
        # some models store preprocessing in attributes
        if scaler is None:
            for attr in ("scaler_", "preprocessor", "transformer"):
                if hasattr(model, attr):
                    candidate = getattr(model, attr)
                    if hasattr(candidate, "transform"):
                        scaler = candidate
                        break
    except Exception:
        scaler = None

# If no scaler found, continue silently (app will align/pad inputs where possible)

# -----------------------------
# 📊 SIDEBAR INPUT
# -----------------------------
st.sidebar.header("🧾 Patient Health Details")

age = st.sidebar.number_input("Age", 1, 100, 30)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 22.0)
glucose = st.sidebar.number_input("Glucose Level", 50.0, 300.0, 100.0)
cholesterol = st.sidebar.number_input("Cholesterol", 100.0, 400.0, 180.0)
hba1c = st.sidebar.number_input("HbA1c Level", 3.0, 15.0, 5.5)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
smoking = st.sidebar.selectbox("Smoking Status", ["Yes", "No"])

# -----------------------------
# 🧠 FEATURE ENGINEERING
# -----------------------------
health_risk_score = glucose + cholesterol + hba1c + bmi

# -----------------------------
# 📦 INPUT DATAFRAME
# -----------------------------
input_data = pd.DataFrame([[
    age, bmi, glucose, cholesterol, hba1c, health_risk_score
]], columns=[
    "age", "bmi", "glucose", "cholesterol", "HbA1c_level", "health_risk_score"
])

# -----------------------------
# 🔘 PREDICTION BUTTON
# -----------------------------
st.markdown("---")

if st.button("🔍 Predict Disease Risk"):
    
    # decide how to feed input to model
    def _get_final_estimator(est):
        if hasattr(est, "named_steps"):
            names = list(est.named_steps.keys())
            return est.named_steps[names[-1]]
        if hasattr(est, "steps"):
            return est.steps[-1][1]
        return est

    final_est = _get_final_estimator(model)

    # prefer letting a pipeline handle preprocessing
    if hasattr(model, "named_steps") or hasattr(model, "steps"):
        model_input = input_data
    else:
        expected_names = getattr(final_est, "feature_names_in_", None)
        expected_n = getattr(final_est, "n_features_in_", None)

        if expected_names is not None:
            # build a full input DataFrame with expected columns, defaulting to 0
            full_df = pd.DataFrame(0, index=range(len(input_data)), columns=expected_names)
            for col in input_data.columns:
                matches = [c for c in expected_names if c.lower() == col.lower()]
                if matches:
                    full_df[matches[0]] = input_data[col].values
            model_input = full_df
            st.info("Input aligned to model feature names where possible; missing features set to 0.")
            if scaler is not None and hasattr(scaler, "transform"):
                try:
                    model_input = scaler.transform(model_input)
                except Exception:
                    st.warning("Scaler failed to transform aligned input; passing raw aligned input to model.")
        elif expected_n is not None:
            # pad or truncate columns to match expected feature count
            if input_data.shape[1] != expected_n:
                diff = expected_n - input_data.shape[1]
                if diff > 0:
                    pad = np.zeros((input_data.shape[0], diff))
                    model_input = np.hstack([input_data.values, pad])
                    st.info(f"Input had {input_data.shape[1]} features; padded with {diff} zeros to match model ({expected_n}).")
                else:
                    model_input = input_data.values[:, :expected_n]
                    st.info(f"Input had more features; truncated to {expected_n} to match model.")
            else:
                model_input = input_data.values
            if scaler is not None and hasattr(scaler, "transform"):
                try:
                    model_input = scaler.transform(model_input)
                except Exception:
                    st.warning("Scaler failed to transform input; passing raw input to model.")
        else:
            model_input = input_data

    # prediction
    prediction = model.predict(model_input)
    try:
        prediction_proba = model.predict_proba(model_input)
    except Exception:
        prediction_proba = None

    st.markdown('<div class="box">', unsafe_allow_html=True)

    # -----------------------------
    # RESULT DISPLAY
    # -----------------------------
    st.subheader("🧠 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Disease Risk Detected")
    else:
        st.success("✅ Low Disease Risk")

    st.write("")

    st.subheader("📊 Prediction Probability")

    if prediction_proba is not None:
        st.write(f"Low Risk: {prediction_proba[0][0]:.2f}")
        st.write(f"High Risk: {prediction_proba[0][1]:.2f}")
    else:
        st.write("Prediction probabilities not available for this model.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 📌 FOOTER
# -----------------------------
st.markdown("---")
st.markdown("👨‍💻 Developed by Ijaz ur Rahman | AI & ML Project")