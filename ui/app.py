import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Resolve absolute path to the model to avoid relative path issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.pkl')

# Load the model
model = joblib.load(MODEL_PATH)

st.title("Heart Disease Prediction")

# Define input fields based on your dataset features (replace with your actual features)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0,1])
restecg = st.selectbox("Resting ECG Results", options=[0,1,2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0,1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0,1,2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0,1,2,3,4])
thal = st.selectbox("Thalassemia", options=[0,1,2,3])

# Create DataFrame from inputs
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Button to predict
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.error(f"Prediction: Heart disease detected (Confidence: {prediction_prob:.2%})")
    else:
        st.success(f"Prediction: No heart disease detected (Confidence: {prediction_prob:.2%})")
