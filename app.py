import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Placement Prediction", layout="centered")
st.title("🎓 Placement Prediction App")

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "pass_pred(logistic_regression)_model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

model = load_model()

# Show expected features (for debugging / learning)
expected_features = list(model.feature_names_in_)

st.subheader("Enter Student Details")

# ---- USER INPUTS ----
ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, 60.0)
hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, 60.0)
degree_p = st.number_input("Degree Percentage", 0.0, 100.0, 60.0)
etest_p = st.number_input("E-Test Percentage", 0.0, 100.0, 60.0)
mba_p = st.number_input("MBA Percentage", 0.0, 100.0, 60.0)

gender = st.selectbox("Gender", ["Male", "Female"])
workex = st.selectbox("Work Experience", ["Yes", "No"])

# Encoding (same logic used in training)
gender = 1 if gender == "Male" else 0
workex = 1 if workex == "Yes" else 0

# ---- CREATE INPUT DATA USING MODEL FEATURES ----
input_dict = {
    "gender": gender,
    "ssc_p": ssc_p,
    "hsc_p": hsc_p,
    "degree_p": degree_p,
    "workex": workex,
    "etest_p": etest_p,
    "mba_p": mba_p
}

# Create DataFrame with EXACT expected columns
input_data = pd.DataFrame([[input_dict[col] for col in expected_features]],
                          columns=expected_features)

# ---- PREDICTION ----
if st.button("Predict Placement"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Student is LIKELY to be Placed  
Probability: {probability:.2%}")
    else:
        st.error(f"❌ Student is NOT Likely to be Placed  
Probability: {probability:.2%}")
