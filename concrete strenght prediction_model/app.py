import streamlit as st
import numpy as np
import joblib

# --- Page Config ---
st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")

# --- Title and Instructions ---
st.title("🧱 Concrete Compressive Strength Predictor")
st.markdown(
    """
    Enter the values for each ingredient in your concrete mixture below.
    The app will predict the compressive strength (in MPa) and classify it as **Low**, **Medium**, or **High** strength.
    """
)

# --- Input Fields ---
st.header("Mixture Features (per m³)")
cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=200.0, step=1.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, value=0.0, step=1.0)
ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0, step=1.0)
water = st.number_input("Water (kg/m³)", min_value=0.0, value=180.0, step=1.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=0.0, step=0.1)
coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, value=900.0, step=1.0)
fine_agg = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, value=800.0, step=1.0)
age = st.number_input("Age (days)", min_value=1, value=28, step=1)

# --- Prediction ---
if st.button("Predict Concrete Strength"):
    # Load model
    model = joblib.load("concrete_prediction")
    # Prepare features
    features = np.array([[cement, slag, ash, water, superplasticizer, coarse_agg, fine_agg, age]])
    # Predict
    strength = float(model.predict(features)[0])

    # --- Categorize Strength ---
    if strength < 20:
        category = "Low Strength"
        color = "red"
        msg = "🔴 **Low Strength:** Suitable for pathways, flooring, or non-load bearing walls."
        st.error(msg)
    elif 20 <= strength < 40:
        category = "Medium Strength"
        color = "orange"
        msg = "🟠 **Medium Strength:** Suitable for general construction (residential buildings, beams, slabs)."
        st.warning(msg)
    else:
        category = "High Strength"
        color = "green"
        msg = "🟢 **High Strength:** Suitable for bridges, high-rise buildings, and heavy-duty construction."
        st.success(msg)

    # --- Show Results ---
    st.subheader("Predicted Compressive Strength")
    st.metric(label="Strength (MPa)", value=f"{strength:.2f}", delta=category)
    st.progress(min(strength / 60, 1.0))  # Assuming 60 MPa as a practical upper bound for progress bar

    # Optional: Show all input values for confirmation
    with st.expander("Show input summary"):
        st.write({
            "Cement": cement,
            "Blast Furnace Slag": slag,
            "Fly Ash": ash,
            "Water": water,
            "Superplasticizer": superplasticizer,
            "Coarse Aggregate": coarse_agg,
            "Fine Aggregate": fine_agg,
            "Age": age
        })
else:
    st.info("Fill in the mixture details and click **Predict Concrete Strength**.")