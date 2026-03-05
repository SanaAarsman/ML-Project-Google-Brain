import os
import joblib
import pandas as pd
import streamlit as st

# -------------------------
# Load model
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models.pkl"))

st.title("🫀 Heart Disease Prediction App")

st.write("Enter patient information below to estimate heart disease risk.")

# -------------------------
# User Inputs
# -------------------------

age = st.slider("Age", 20, 90, 50)

sex_ui = st.selectbox("Sex", ["Male", "Female"])
sex = "M" if sex_ui == "Male" else "F"

chest_ui = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

chest_map = {
    "Typical Angina": "TA",
    "Atypical Angina": "ATA",
    "Non-Anginal Pain": "NAP",
    "Asymptomatic": "ASY"
}

chest_pain = chest_map[chest_ui]

resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)

cholesterol = st.number_input("Cholesterol", 100, 600, 200)

fasting_ui = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fasting_bs = 1 if fasting_ui == "Yes" else 0

resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)

exercise_ui = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exercise_angina = "Y" if exercise_ui == "Yes" else "N"

oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

st_slope = st.selectbox("ST Segment Slope", ["Up", "Flat", "Down"])

# -------------------------
# Prediction
# -------------------------

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    # Debug: show the data sent to model
    st.subheader("Input Data")
    st.write(input_data)

    prediction = model.predict(input_data)[0]

    # Probability
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("⚠️ Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    if probability is not None:
        st.write(f"Risk Probability: **{probability*100:.2f}%**")
        st.progress(float(probability))