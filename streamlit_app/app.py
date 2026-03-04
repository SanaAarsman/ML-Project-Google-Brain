import os
from flask import app
import joblib
import pandas as pd
import streamlit as st



st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models.pkl")
DATA_PATH = os.path.join(BASE_DIR, "heart_disease.csv")


# -----------------------------
# Load model (Pipeline: preprocessing + classifier)
# -----------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = load_model(MODEL_PATH)


# Optional: load dataset only to set sensible defaults
@st.cache_data
def load_data(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data(DATA_PATH)


def default_for(col: str, fallback):
    """Use dataset median/mode as default if CSV exists, otherwise fallback."""
    if df is None or col not in df.columns:
        return fallback
    s = df[col].dropna()
    if s.empty:
        return fallback
    if pd.api.types.is_numeric_dtype(s):
        return float(s.median())
    return str(s.mode().iloc[0])


st.title("🫀 Heart Disease Prediction")
st.write("Fill in the patient information below and click **Predict**.")


# -----------------------------
# Input form
# -----------------------------
with st.form("prediction_form"):
    st.subheader("Patient data")

    col1, col2, col3 = st.columns(3)

    # Numeric inputs
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120,
                              value=int(default_for("Age", 40)), step=1)
        resting_bp = st.number_input("RestingBP", min_value=0, max_value=300,
                                     value=int(default_for("RestingBP", 120)), step=1)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700,
                                      value=int(default_for("Cholesterol", 200)), step=1)
        max_hr = st.number_input("MaxHR", min_value=0, max_value=250,
                                 value=int(default_for("MaxHR", 150)), step=1)

    with col2:
        fasting_bs = st.selectbox("FastingBS (0/1)", options=[0, 1],
                                  index=0 if int(default_for("FastingBS", 0)) == 0 else 1)
        oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=10.0,
                                  value=float(default_for("Oldpeak", 0.0)), step=0.1)

    # Categorical inputs (match dataset categories)
    with col3:
        sex = st.selectbox("Sex", options=["M", "F"],
                           index=0 if default_for("Sex", "M") == "M" else 1)

        chest_pain_type = st.selectbox(
            "ChestPainType",
            options=["ASY", "ATA", "NAP", "TA"],
            index=["ASY", "ATA", "NAP", "TA"].index(default_for("ChestPainType", "ASY"))
            if default_for("ChestPainType", "ASY") in ["ASY", "ATA", "NAP", "TA"] else 0
        )

        resting_ecg = st.selectbox(
            "RestingECG",
            options=["Normal", "ST", "LVH"],
            index=["Normal", "ST", "LVH"].index(default_for("RestingECG", "Normal"))
            if default_for("RestingECG", "Normal") in ["Normal", "ST", "LVH"] else 0
        )

        exercise_angina = st.selectbox(
            "ExerciseAngina",
            options=["N", "Y"],
            index=0 if default_for("ExerciseAngina", "N") == "N" else 1
        )

        st_slope = st.selectbox(
            "ST_Slope",
            options=["Up", "Flat", "Down"],
            index=["Up", "Flat", "Down"].index(default_for("ST_Slope", "Flat"))
            if default_for("ST_Slope", "Flat") in ["Up", "Flat", "Down"] else 1
        )

    submitted = st.form_submit_button("Predict")


if submitted:
    # Build a single-row dataframe with EXACT column names used in training
    X_input = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])

    try:
        pred = model.predict(X_input)[0]

        # Some models/pipelines support predict_proba
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf", None), "predict_proba"):
            proba = model.named_steps["clf"].predict_proba(model.named_steps["preprocess"].transform(X_input))[0]

        st.markdown("---")
        st.subheader("Result")

        if pred == 1:
            st.error("Prediction: **Heart Disease (1)**")
        else:
            st.success("Prediction: **No Heart Disease (0)**")

        if proba is not None and len(proba) == 2:
            st.write(f"Probability of Heart Disease (class 1): **{proba[1]*100:.1f}%**")
            st.progress(float(proba[1]))

        with st.expander("Show input data"):
            st.dataframe(X_input)

    except Exception as e:
        st.error("Something went wrong during prediction.")
        st.exception(e)
    
 