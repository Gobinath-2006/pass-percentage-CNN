import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Placement Prediction", layout="centered")
st.title("🎓 Placement Prediction App")

# ---------- LOAD MODEL ----------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "pass_pred(logistic_regression)_model.pkl"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Upload the .pkl file to GitHub root.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------- INPUT UI ----------
st.subheader("Enter Student Details")

gender = st.selectbox("Gender", ["Male", "Female"])
ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, 60.0)
hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, 60.0)
degree_p = st.number_input("Degree Percentage", 0.0, 100.0, 60.0)
etest_p = st.number_input("E-Test Percentage", 0.0, 100.0, 60.0)
mba_p = st.number_input("MBA Percentage", 0.0, 100.0, 60.0)
workex = st.selectbox("Work Experience", ["Yes", "No"])

# ---------- ENCODING ----------
gender = 1 if gender == "Male" else 0
workex = 1 if workex == "Yes" else 0

# ---------- CREATE INPUT (MATCH MODEL FEATURES) ----------
feature_names = model.feature_names_in_

input_values = {
    "gender": gender,
    "ssc_p": ssc_p,
    "hsc_p": hsc_p,
    "degree_p": degree_p,
    "workex": workex,
    "etest_p": etest_p,
    "mba_p": mba_p
}

input_data = pd.DataFrame(
    [[input_values[col] for col in feature_names]],
    columns=feature_names
)

# ---------- PREDICTION ----------
if st.button("Predict Placement"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"✅ Student is LIKELY to be Placed\n\nProbability: {prob:.2%}")
    else:
        st.error(f"❌ Student is NOT Likely to be Placed\n\nProbability: {prob:.2%}")
