# Bank Term Deposit Subscription Predictor

Predicts whether a bank customer will subscribe to a term deposit, to help
prioritize who a marketing team should call.

**Live app:** https://bank-subscription-predictor-6f7e87vwrf49dpupr7ffzz.streamlit.app/

## Overview

This project trains a classifier on the UCI Bank Marketing dataset and ships
it as an interactive Streamlit app. Given a customer's demographics, financial
info, and contact history, it predicts subscription likelihood and shows
which factors drove that specific prediction.

## Dataset

`bank.csv` — Bank Marketing dataset (11k+ customers), with demographic,
financial, and campaign interaction features. [UCI Machine Learning
Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## What's in this repo

- **`bank_subscription_analysis.ipynb`** — data exploration, preprocessing,
  and model training (Logistic Regression vs. Random Forest)
- **`app.py`** — the Streamlit app: takes all 41 model features as input and
  returns a prediction plus a SHAP-based explanation of the top drivers
- **`rf_model.zip`** — the trained Random Forest model (unzipped automatically
  by the app)
- **`model_features.pkl`** — the exact feature order the model expects

## Key steps

- Removed `duration` before training — it's only known *after* a call
  happens, so keeping it in would leak the outcome into the features
- Compared Logistic Regression against Random Forest
- A preliminary fairness check compares subscription rates across marital
  status groups as a sanity check for bias (not a full disparate-impact
  audit — see Limitations)

## Model performance

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.77 |
| Random Forest | 0.78 |

(Both scores are after removing `duration` to eliminate leakage.)

## Running locally

```bash
git clone https://github.com/teja-107/Bank-Subscription-Predictor.git
cd Bank-Subscription-Predictor
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, Pandas, Scikit-learn, SHAP, Streamlit, Matplotlib

## Limitations / next steps

- The fairness check is a single-group rate comparison, not a full
  disparate-impact analysis across all protected attributes
- The notebook doesn't yet include the step that saves the final model to
  `rf_model.pkl` / `model_features.pkl` — that's currently done separately
- No automated tests yet
