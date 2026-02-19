# heart_streamlit_app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -----------------------------
# Load trained model & scaler
# -----------------------------
# Make sure you saved your model and scaler in Colab like:
# pickle.dump(model, open("heart_model.pkl", "wb"))
# pickle.dump(scaler, open("scaler.pkl", "wb"))

model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load dataset just to get column names (optional)
df = pd.read_csv("health_data.csv")
X_columns = df.drop("cardio", axis=1).columns


# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient information:")

# Create input fields dynamically
user_input = []
for col in X_columns:
    val = st.number_input(f"{col}", value=int(df[col].median()))
    user_input.append(val)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=X_columns)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 0:
        st.success("✅ The person is healthy")
    else:
        st.error("⚠️ The person has a risk of heart disease")

# Optional: show model accuracy if you want
if st.checkbox("Show Model Accuracy"):
    X = df.drop("cardio", axis=1)
    y = df["cardio"]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    acc = np.mean(y_pred == y)
    st.write(f"Model Accuracy: {acc:.2f}")
