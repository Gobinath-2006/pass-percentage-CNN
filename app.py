import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Placement Prediction", layout="centered")
st.title("🎓 Placement Prediction App")

# -------------------------------
# Load Logistic Regression Model
# -------------------------------
@st.cache_resource
def load_model():
    with open("pass_pred(logistic_regression)_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
st.success("✅ Model loaded successfully")

# -------------------------------
# Load Dataset (MANDATORY)
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Placement_Data_Full_Class.csv")

df = load_data()
st.success("✅ Dataset loaded successfully")

# -------------------------------
# Prepare Features
# -------------------------------
target_col = "status"   # Placed / Not Placed
X = df.drop(columns=[target_col])

st.header("Enter Student Details")

user_input = {}

for col in X.columns:
    if X[col].dtype == "object":
        user_input[col] = st.selectbox(col, X[col].unique())
    else:
        user_input[col] = st.number_input(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

input_df = pd.DataFrame([user_input])

# -------------------------------
# Encode Categorical Columns
# -------------------------------
full_df = pd.concat([input_df, X], axis=0)
full_df = pd.get_dummies(full_df)
input_df = full_df.iloc[:1]

# Align with model features
input_df = input_df.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Placement"):
    result = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if result == 1:
        st.success(f"🎉 Student WILL be Placed (Confidence: {prob:.2%})")
    else:
        st.error(f"❌ Student will NOT be Placed (Confidence: {1-prob:.2%})")
