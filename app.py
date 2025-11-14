import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# LOAD MODEL AND FEATURES
# -----------------------------
@st.cache_resource
def load_model():
    with open("risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# FEATURES IN EXACT ORDER FOR MODEL
feature_cols = [
    "open", "high", "low", "close", "adj_close", "volume",
    "return", "spread", "return_5d", "volume_5d",
    "sma_5", "sma_10", "sma_ratio", "momentum_10",
    "range", "return_lag1", "return_lag2",
    "volatility_60"
]

st.title("Investment Risk Predictor")
st.write("Enter just 5 values ​​- everything else is calculated automatically.")

open_p = st.number_input("Open Price", value=100.0)
high_p = st.number_input("High Price", value=101.0)
low_p = st.number_input("Low Price", value=99.0)
close_p = st.number_input("Close Price", value=100.5)
volume_p = st.number_input("Volume", value=1_000_000)

ret = (close_p - open_p) / open_p if open_p != 0 else 0
spread = close_p - open_p

return_5d = 0       
volume_5d = volume_p  

sma_5 = 0
sma_10 = 0
sma_ratio = 0

momentum_10 = 0    

range_ = high_p - low_p

return_lag1 = 0     
return_lag2 = 0

volatility_60 = 0   


X = pd.DataFrame([[
    open_p, high_p, low_p, close_p, close_p, volume_p,
    ret, spread, return_5d, volume_5d,
    sma_5, sma_10, sma_ratio, momentum_10,
    range_, return_lag1, return_lag2,
    volatility_60
]], columns=feature_cols)


if st.button("Predict Risk Level"):
    pred = model.predict(X)[0]

    if pred == 0:
        st.success("LOW RISK (0)")
    elif pred == 1:
        st.warning("MEDIUM RISK (1)")
    else:
        st.error("HIGH RISK (2)")
