import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# ------------------ BACKGROUND FUNCTION ------------------
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded = base64.b64encode(file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                background-repeat: no-repeat;
            }}

            .main {{
                background-color: rgba(255, 255, 255, 0.85);
                padding: 2rem;
                border-radius: 10px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Background image not found. Skipping background.")

add_bg_from_local("bg.jpg")

# ------------------ LOAD MODEL ------------------
try:
    model = joblib.load("churn_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ------------------ TITLE ------------------
st.markdown("<h1 style='text-align: center;'>Customer Churn Predictor</h1>", unsafe_allow_html=True)

# ------------------ INPUT UI ------------------
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, step=1.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, step=10.0)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.markdown("---")

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(
    gender, senior_citizen, partner, dependents, phone_service, multiple_lines,
    tenure, monthly_charges, total_charges, internet_service,
    online_security, online_backup, device_protection, tech_support,
    streaming_tv, streaming_movies, contract, paperless_billing,
    payment_method, model_columns
):

    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Align with model columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    return input_df

# ------------------ PREDICTION ------------------
if st.button("Predict Churn"):

    # Input validation
    if total_charges == 0 and tenure > 0:
        st.warning("⚠️ Total charges seem inconsistent with tenure.")

    input_data = preprocess_input(
        gender, senior_citizen, partner, dependents, phone_service, multiple_lines,
        tenure, monthly_charges, total_charges, internet_service,
        online_security, online_backup, device_protection, tech_support,
        streaming_tv, streaming_movies, contract, paperless_billing,
        payment_method, model_columns
    )

    with st.spinner("Predicting churn..."):
        probability = model.predict_proba(input_data)[0][1]

    threshold = 0.5
    prediction = 1 if probability >= threshold else 0

    # Debug info
    st.write("🔍 Raw Churn Probability:", probability)
    st.write("🔍 Threshold Used:", threshold)
    st.write("🔍 Prediction Output:", prediction)

    # Result
    st.markdown("### Prediction Result")
    st.metric(label="Churn Probability", value=f"{probability * 100:.2f}%")

    if prediction == 1:
        st.markdown(
            f"""
            <div style="background-color:#ffe6e6;padding:20px;border-radius:10px;">
                <h3 style="color:#cc0000;">This customer is <strong>likely to churn</strong>.</h3>
                <p>Confidence: <strong>{probability * 100:.2f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#e6ffe6;padding:20px;border-radius:10px;">
                <h3 style="color:#006600;">This customer is <strong>likely to stay</strong>.</h3>
                <p>Confidence: <strong>{(1 - probability) * 100:.2f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Risk interpretation
    if probability > 0.7:
        st.warning("🚨 High churn risk – immediate action recommended.")
    elif probability > 0.4:
        st.info("⚠️ Moderate churn risk.")
    else:
        st.success("✅ Low churn risk.")

# ------------------ INPUT SUMMARY ------------------
st.markdown("---")
st.subheader("Input Summary")

raw_input_dict = {
    "Gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure (Months)": tenure,
    "Total Charges": total_charges,
    "Contract Type": contract,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method
}

raw_input_df = pd.DataFrame([raw_input_dict])
st.dataframe(raw_input_df)