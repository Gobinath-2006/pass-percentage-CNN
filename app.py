import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Placement Prediction", layout="centered")

st.title("🎓 Placement Prediction App")

# Absolute path to model
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "pass_pred(logistic_regression)_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please upload the .pkl file to GitHub root.")
        st.stop()
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

model = load_model()

st.subheader("Enter Student Details")

ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, 60.0)
hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, 60.0)
degree_p = st.number_input("Degree Percentage", 0.0, 100.0, 60.0)
etest_p = st.number_input("E-Test Percentage", 0.0, 100.0, 60.0)
mba_p = st.number_input("MBA Percentage", 0.0, 100.0, 60.0)

gender = st.selectbox("Gender", ["Male", "Female"])
workex = st.selectbox("Work Experience", ["Yes", "No"])

# Encoding (same as training)
gender = 1 if gender == "Male" else 0
workex = 1 if workex == "Yes" else 0

input_data = pd.DataFrame([[
    gender,
    ssc_p,
    hsc_p,
    degree_p,
    workex,
    etest_p,
    mba_p
]], columns=[
    "gender",
    "ssc_p",
    "hsc_p",
    "degree_p",
    "workex",
    "etest_p",
    "mba_p"
])

if st.button("Predict Placement"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Student is LIKELY to be Placed")
    else:
        st.error("❌ Student is NOT Likely to be Placed")

