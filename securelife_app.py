import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load('Insurance_premium_model.pkl')

st.set_page_config(page_title="Insurance Premium Estimator", layout="centered")

# App Header
st.title("🏦 Insurance Premium Estimator")
st.markdown("### 💡 Predict your expected insurance premium using ML.")
st.write("Fill in the details below:")

# Input Widgets
age = st.slider("Age", 18, 90, 30)
annual_income = st.number_input("Annual Income (₦)", min_value=100000.0, value=500000.0)
health_score = st.slider("Health Score", 1, 100, 75)
years_since_policy = st.number_input("Years Since Policy Start", min_value=0, value=5)

# Sample categorical inputs
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
policy_type = st.selectbox("Policy Type", ["Comprehensive", "Third Party", "Life", "Health"])

# Encode categorical inputs manually
gender_encoded = 1 if gender == "Male" else 0
marital_encoded = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2,
    "Widowed": 3
}[marital_status]
policy_type_encoded = {
    "Comprehensive": 0,
    "Third Party": 1,
    "Life": 2,
    "Health": 3
}[policy_type]

# Prepare input for model (match training format)
input_data = pd.DataFrame({
    'Age': [age],
    'Annual Income': [np.log1p(annual_income)],
    'Health Score': [np.log1p(health_score)],
    'Years Since Policy Start': [years_since_policy],
    'Gender': [gender_encoded],
    'Marital Status': [marital_encoded],
    'Policy Type': [policy_type_encoded]
})

# Predict Button
if st.button("🔮 Predict Premium"):
    prediction_log = model.predict(input_data)[0]
    premium = np.expm1(prediction_log)
    st.success(f"💰 Estimated Premium: ₦{round(premium, 2):,}")

    # Optional Residual Plot Placeholder
    st.markdown("### 📊 Prediction Summary")
    st.dataframe(input_data)