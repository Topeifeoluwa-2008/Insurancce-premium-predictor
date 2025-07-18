import streamlit as st
import numpy as np
import pandas as pd
import joblib
from textblob import TextBlob

# Load saved model and preprocessing tools
model = joblib.load('model/Insurance_premium_model.pkl')
imputer = joblib.load('model/imputer.pkl')
training_columns = joblib.load('model/training_columns.pkl')

# Set Streamlit page config
st.set_page_config(page_title="Insurance Premium Estimator", layout="centered")

# App Header
st.title("🏦 Insurance Premium Estimator")
st.markdown("### 💡 Predict your expected insurance premium using ML.")
st.write("Fill in the details below:")

# Numeric Inputs
age = st.slider("Age", 18, 90, 30)
annual_income = st.number_input("Annual Income (₦)", min_value=100000.0, value=500000.0)
health_score = st.slider("Health Score", 1, 100, 75)
years_since_policy = st.number_input("Years Since Policy Start", min_value=0, value=5)

# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education = st.selectbox("Education Level", ["Primary", "Secondary", "Tertiary"])
occupation = st.selectbox("Occupation", ["Unemployed", "Engineer", "Teacher", "Doctor", "Other"])
location = st.selectbox("Location", ["Lagos", "Abuja", "Kano", "Other"])
policy_type = st.selectbox("Policy Type", ["Comprehensive", "Third Party", "Life", "Health"])
smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
exercise_freq = st.selectbox("Exercise Frequency", ["Rarely", "Occasionally", "Regularly"])
property_type = st.selectbox("Property Type", ["House", "Apartment", "Other"])
feedback = st.text_area("Customer Feedback", "Great service and quick response!")

# Predict Button
if st.button("🔮 Predict Premium"):
    # Build input dictionary
    input_dict = {
        'Age': [age],
        'Annual Income': [annual_income],
        'Health Score': [health_score],
        'Years Since Policy Start': [years_since_policy],
        'Gender': [gender],
        'Marital Status': [marital_status],
        'Education Level': [education],
        'Occupation': [occupation],
        'Location': [location],
        'Policy Type': [policy_type],
        'Smoking Status': [smoking_status],
        'Exercise Frequency': [exercise_freq],
        'Property Type': [property_type],
        'Customer Feedback': [feedback],
        'Policy Start Date': ['2023-01-01'],  
        'Premium Amount': [0]  
    }

    df_input = pd.DataFrame(input_dict)

    # Add sentiment scores
    df_input['Feedback_Polarity'] = df_input['Customer Feedback'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_input['Feedback_Subjectivity'] = df_input['Customer Feedback'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

    # Drop unnecessary columns
    df_input = df_input.drop(['Customer Feedback', 'Policy Start Date', 'Premium Amount'], axis=1)

    # One-hot encode categorical columns
    cat_cols = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
                'Policy Type', 'Smoking Status', 'Exercise Frequency', 'Property Type']
    df_encoded = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)

    # Align input columns with training features
    df_encoded = df_encoded.reindex(columns=training_columns, fill_value=0)

    # Impute missing values
    input_imputed = imputer.transform(df_encoded)

    # Predict
    prediction = model.predict(input_imputed)[0]
    st.success(f"💰 Estimated Premium: ₦{round(prediction, 2):,}")
