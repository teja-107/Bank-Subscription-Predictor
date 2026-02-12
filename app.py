import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import zipfile

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Bank Subscription Predictor", layout="wide")

# -------------------------
# Ensure model exists (unzip if needed)
# -------------------------
if not os.path.exists("rf_model.pkl") and os.path.exists("rf_model.zip"):
    with zipfile.ZipFile("rf_model.zip", "r") as zip_ref:
        zip_ref.extractall()

# -------------------------
# Load model + features
# -------------------------
model = joblib.load("rf_model.pkl")
model_features = joblib.load("model_features.pkl")

# -------------------------
# SHAP Tree Explainer (stable for deployment)
# -------------------------
explainer = shap.TreeExplainer(model)

# -------------------------
# Title
# -------------------------
st.title("🏦 Bank Term Deposit Subscription Predictor")
st.write("Predict whether a customer will subscribe and understand why.")

st.divider()

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age", 18, 100, 35)
balance = st.sidebar.slider("Account Balance", 0, 100000, 1000)
campaign = st.sidebar.slider("Campaign Contacts", 0, 50, 1)
pdays = st.sidebar.slider("Days Since Last Contact", -1, 500, -1)
previous = st.sidebar.slider("Previous Contacts", 0, 50, 0)

housing_yes = st.sidebar.selectbox("Housing Loan", [0, 1])
loan_yes = st.sidebar.selectbox("Personal Loan", [0, 1])

# -------------------------
# Predict
# -------------------------
if st.sidebar.button("Predict Subscription"):

    input_data = pd.DataFrame([[0]*len(model_features)], columns=model_features)

    if 'age' in input_data.columns:
        input_data['age'] = age
    if 'balance' in input_data.columns:
        input_data['balance'] = balance
    if 'campaign' in input_data.columns:
        input_data['campaign'] = campaign
    if 'pdays' in input_data.columns:
        input_data['pdays'] = pdays
    if 'previous' in input_data.columns:
        input_data['previous'] = previous
    if 'housing_yes' in input_data.columns:
        input_data['housing_yes'] = housing_yes
    if 'loan_yes' in input_data.columns:
        input_data['loan_yes'] = loan_yes

    # -------------------------
    # Prediction
    # -------------------------
    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Customer likely to SUBSCRIBE ✅")
    else:
        st.error("Customer unlikely to subscribe ❌")

    st.metric("Subscription Probability", f"{prob:.2%}")

    st.divider()

    # -------------------------
    # SHAP explanation (stable)
    # -------------------------
    st.subheader("🔍 Why did the model predict this?")

    shap_values = explainer(input_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)

    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    st.pyplot(fig)





