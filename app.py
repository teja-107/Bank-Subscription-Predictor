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
# Load model + features (cached so they don't reload on every interaction)
# -------------------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load("rf_model.pkl")
    model_features = joblib.load("model_features.pkl")
    explainer = shap.TreeExplainer(model)
    return model, model_features, explainer

model, model_features, explainer = load_model_and_features()

# -------------------------
# Title
# -------------------------
st.title("🏦 Bank Term Deposit Subscription Predictor")
st.write("Predict whether a customer will subscribe and understand why.")

st.divider()

# -------------------------
# Sidebar inputs — every feature the model was trained on
# -------------------------
st.sidebar.header("Customer Information")

# --- Numeric features ---
st.sidebar.subheader("Demographics & Contact History")
age = st.sidebar.slider("Age", 18, 100, 35)
balance = st.sidebar.slider("Account Balance", -8000, 100000, 1000)
day = st.sidebar.slider("Day of Month Last Contacted", 1, 31, 15)
campaign = st.sidebar.slider("Contacts During This Campaign", 0, 50, 1)
pdays = st.sidebar.slider("Days Since Last Contact (-1 = never contacted)", -1, 500, -1)
previous = st.sidebar.slider("Contacts Before This Campaign", 0, 50, 0)

# --- Categorical features (dropdowns, matching the training data's categories) ---
st.sidebar.subheader("Personal Details")
job = st.sidebar.selectbox(
    "Job",
    ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
     "retired", "self-employed", "services", "student", "technician",
     "unemployed", "unknown"],
)
marital = st.sidebar.selectbox("Marital Status", ["divorced", "married", "single"])
education = st.sidebar.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.sidebar.selectbox("Has Credit in Default?", ["no", "yes"])
housing = st.sidebar.selectbox("Has Housing Loan?", ["no", "yes"])
loan = st.sidebar.selectbox("Has Personal Loan?", ["no", "yes"])

st.sidebar.subheader("Campaign Details")
contact = st.sidebar.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
month = st.sidebar.selectbox(
    "Last Contact Month",
    ["apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar", "may", "nov", "oct", "sep"],
)
poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", ["failure", "other", "success", "unknown"])

# -------------------------
# Predict
# -------------------------
if st.sidebar.button("Predict Subscription"):

    # Start with every model feature at 0 (this is what get_dummies(..., drop_first=True)
    # produces for the "baseline" category of each categorical column)
    input_data = pd.DataFrame([[0] * len(model_features)], columns=model_features)

    # --- Numeric values go straight in ---
    numeric_values = {
        "age": age,
        "balance": balance,
        "day": day,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
    }
    for col, val in numeric_values.items():
        if col in input_data.columns:
            input_data[col] = val

    # --- Categorical values: flip the matching one-hot column to 1 ---
    # (if the selection equals the dropped baseline category, every dummy
    # column for that feature correctly stays 0 — that IS the baseline)
    categorical_selections = {
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
    }
    for prefix, selection in categorical_selections.items():
        dummy_col = f"{prefix}_{selection}"
        if dummy_col in input_data.columns:
            input_data[dummy_col] = 1

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

    # -------------------------
    # SHAP explanation
    # -------------------------
    st.divider()
    st.subheader("🔍 Model Explanation")

    shap_values = explainer.shap_values(input_data)

    # For a binary RandomForestClassifier, shap_values is a list [class_0, class_1]
    values_for_class_1 = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    contrib = (
        pd.Series(values_for_class_1, index=model_features)
        .sort_values(key=abs, ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in contrib.values]
    ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
    ax.set_xlabel("Impact on subscription probability")
    ax.set_title("Top 10 features driving this prediction")
    st.pyplot(fig)

    st.caption(
        "Green bars push the prediction toward 'Subscribe', red bars push it toward 'Not Subscribe'. "
        "Bar length shows the size of that feature's impact on this specific prediction."
    )
