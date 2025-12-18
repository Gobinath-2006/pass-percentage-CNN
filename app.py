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

# ---------- USER INPUT ----------
st.subheader("Enter Student Details")

gender = st.selectbox("Gender", ["Male", "Female"])
workex = st.selectbox("Work Experience", ["Yes", "No"])

ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, 60.0)
hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, 60.0)
degree_p = st.number_input("Degree Percentage", 0.0, 100.0, 60.0)
etest_p = st.number_input("E-Test Percentage", 0.0, 100.0, 60.0)
mba_p = st.number_input("MBA Percentage", 0.0, 100.0, 60.0)

# ---------- CREATE EMPTY INPUT WITH ALL FEATURES ----------
feature_names = model.feature_names_in_
input_data = pd.DataFrame(0, index=[0], columns=feature_names)

# ---------- FILL NUMERIC FEATURES ----------
for col in ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]:
    if col in input_data.columns:
        input_data[col] = locals()[col]

# ---------- HANDLE ENCODED CATEGORICAL FEATURES ----------
if f"gender_{gender}" in input_data.columns:
    input_data[f"gender_{gender}"] = 1
elif "gender" in input_data.columns:
    input_data["gender"] = 1 if gender == "Male" else 0

if f"workex_{workex}" in input_data.columns:
    input_data[f"workex_{workex}"] = 1
elif "workex" in input_data.columns:
    input_data["workex"] = 1 if workex == "Yes" else 0

# ---------- PREDICTION ----------
if st.button("Predict Placement"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Student is LIKELY to be Placed\n\nProbability: {probability:.2%}")
    else:
        st.error(f"❌ Student is NOT Likely to be Placed\n\nProbability: {probability:.2%}")
