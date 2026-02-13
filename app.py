import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("heart_model.pkl", "rb"))

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient information:")

age = st.number_input("Age", 1, 120)

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])

thalach = st.number_input("Max Heart Rate Achieved")

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.number_input("ST Depression (oldpeak)", format="%.1f")

slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])

ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

if st.button("Predict"):
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")
