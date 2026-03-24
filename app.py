import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title("💳 Credit Card Fraud Detection")

# Load model
model = joblib.load("fraud_model.pkl")

# Load dataset structure
df = pd.read_csv("creditcard.csv")

# Prepare scaler exactly like training
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
df["scaled_time"] = scaler.fit_transform(df[["Time"]])

# Drop original columns
df = df.drop(["Amount", "Time", "Class"], axis=1)

st.write("Modify transaction values below:")

sample = df.iloc[0].copy()

input_data = {}

for col in df.columns:
    input_data[col] = st.number_input(col, value=float(sample[col]))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.metric("Fraud Probability", f"{probability:.4f}")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")
