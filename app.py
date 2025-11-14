import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# LOAD MODEL + FEATURES
# -----------------------------
@st.cache_resource
def load_model():
    with open("risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# FEATURES IN EXACT ORDER:
feature_cols = [
    "open", "high", "low", "close", "adj_close", "volume",
    "return", "spread", "return_5d", "volume_5d",
    "sma_5", "sma_10", "sma_ratio", "momentum_10",
    "range", "return_lag1", "return_lag2",
    "volatility_60"
]

st.title("📈 Investment Risk Predictor")

st.write("Введите все 18 параметров, модель обучена именно на них:")

inputs = {}

for col in feature_cols:
    inputs[col] = st.number_input(col, value=0.0, format="%.6f")

# -----------------------------
# PREDICT
# -----------------------------
if st.button("Predict Risk Level"):
    X = pd.DataFrame([[inputs[col] for col in feature_cols]], columns=feature_cols)

    pred = model.predict(X)[0]

    if pred == 0:
        st.success("🟢 Риск низкий (0)")
    elif pred == 1:
        st.warning("🟡 Риск средний (1)")
    else:
        st.error("🔴 Риск высокий (2)")
