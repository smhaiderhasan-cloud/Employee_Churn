import pandas as pd
import streamlit as st
import pickle

with open('employee_churn_schema.pkl', 'rb') as file:
    schema = pickle.load(file)

with open('employee_churn_pipeline.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

st.title('Employee Churn Predictor')
st.subheader('Created by Haider')

education = st.selectbox('Education', ['Bachelors', 'Masters', 'PHD'])
joining_year = st.number_input('Joining Year', min_value=2012, max_value=2026, step=1)
city = st.selectbox('City', ['Bangalore', 'Pune', 'New Delhi'])
payment_tier = st.selectbox('Payment Tier', [1, 2, 3])
age = st.number_input('Age', min_value=18, max_value=65, step=1)
gender = st.selectbox('Gender', ['Male', 'Female'])
ever_benched = st.selectbox('Ever Benched', ['Yes', 'No'])
experience = st.number_input('Experience in Current Domain', min_value=0, max_value=10, step=1)

input_data = pd.DataFrame([{
    'Education': education,
    'JoiningYear': joining_year,
    'City': city,
    'PaymentTier': payment_tier,
    'Age': age,
    'Gender': gender,
    'EverBenched': ever_benched,
    'ExperienceInCurrentDomain': experience
}])

if st.button('Predict'):
    prediction = model_pipeline.predict(input_data)[0]

    if prediction == 1:
        st.error('Prediction: Employee is likely to leave')
    else:
        st.success('Prediction: Employee is likely to stay')