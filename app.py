import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# === Load Model and Encoders ===
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('lable_encoder_gender.pkl', 'rb') as file:  # ✅ as you named it
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# === Streamlit Title ===
st.title("💼 Customer Churn Prediction")

# === User Inputs ===
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.selectbox('Number of Products', [1, 2, 3, 4])
has_cr_card = st.selectbox('Has Credit Card (0 = No, 1 = Yes)', [0, 1])
is_active_member = st.selectbox('Is Active Member (0 = No, 1 = Yes)', [0, 1])
geo_location = st.selectbox('Geography', ['France', 'Germany', 'Spain'])

# === Encode Inputs ===
gender_encoded = label_encoder_gender.transform([gender])[0]
geo_encoded = onehot_encoder_geo.transform([[geo_location]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# === Prepare Input Data ===
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Add geo columns
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# === Match Columns Exactly with Scaler ===
expected_cols = list(scaler.feature_names_in_)
actual_cols = list(input_data.columns)

if expected_cols != actual_cols:
    st.error("❌ Column mismatch with scaler!")
    st.write("**Expected Columns:**", expected_cols)
    st.write("**Provided Columns:**", actual_cols)
    st.stop()

# === Scale and Predict ===
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# === Show Result ===
st.write(f'🔍 **Churn Probability:** {prediction_proba:.2f}')
if prediction_proba > 0.5:
    st.error('⚠️ The customer is **likely to churn**.')
else:
    st.success('✅ The customer is **not likely to churn**.')
