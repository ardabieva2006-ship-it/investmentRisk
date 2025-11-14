import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# -----------------------------
# LOGO
# -----------------------------
def add_logo():
    with open("logo.png", "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-top: -30px;">
            <img src="data:image/png;base64,{encoded}" width="220">
        </div>
        """,
        unsafe_allow_html=True
    )

add_logo()

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    with open("risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# FEATURES
feature_cols = [
    "open", "high", "low", "close", "adj_close", "volume",
    "return", "spread", "return_5d", "volume_5d",
    "sma_5", "sma_10", "sma_ratio", "momentum_10",
    "range", "return_lag1", "return_lag2",
    "volatility_60"
]

# -----------------------------
# CSS
# -----------------------------
green_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #D0F0C0;
}
[data-testid="stSidebar"] {
    background-color: #d4f5df;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
.stButton>button {
    background-color: #1faa59;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #178f49;
}
</style>
"""
st.markdown(green_css, unsafe_allow_html=True)

# -----------------------------
# FEATURE ENGINEERING (как в обучении)
# -----------------------------
def compute_features(open_p, high_p, low_p, close_p, volume_p):

    # создаём временной ряд (эмуляция тикера)
    adj = np.linspace(open_p * 0.9, close_p, 70)

    df = pd.DataFrame({
        "adj_close": adj,
        "close": adj,
        "open": adj * 0.98,
        "high": adj * 1.05,
        "low": adj * 0.95,
        "volume": np.linspace(volume_p*0.7, volume_p, 70)
    })

    # заменяем последний день пользовательскими значениями
    df.loc[len(df)-1, "open"] = open_p
    df.loc[len[df)-1, "high"] = high_p
    df.loc[len(df)-1, "low"] = low_p
    df.loc[len(df)-1, "close"] = close_p
    df.loc[len(df)-1, "adj_close"] = close_p
    df.loc[len(df)-1, "volume"] = volume_p

    # ---- ФИЧИ как в обучении ----
    df["return"] = df["adj_close"].pct_change()
    df["sma_5"] = df["adj_close"].rolling(5).mean()
    df["sma_10"] = df["adj_close"].rolling(10).mean()
    df["sma_ratio"] = df["sma_5"] / df["sma_10"]
    df["momentum_10"] = df["adj_close"].diff(10)
    df["range"] = df["high"] - df["low"]
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)
    df["volatility_60"] = df["return"].rolling(60).std()
    df["spread"] = (df["high"] - df["low"]) / df["close"]
    df["return_5d"] = df["adj_close"].pct_change(5)
    df["volume_5d"] = df["volume"].pct_change(5)

    df = df.dropna().reset_index(drop=True)
    last = df.iloc[-1]

    return pd.DataFrame([[
        last["open"], last["high"], last["low"], last["close"], last["adj_close"], last["volume"],
        last["return"], last["spread"], last["return_5d"], last["volume_5d"],
        last["sma_5"], last["sma_10"], last["sma_ratio"], last["momentum_10"],
        last["range"], last["return_lag1"], last["return_lag2"], last["volatility_60"]
    ]], columns=feature_cols)

# -----------------------------
# UI INPUTS
# -----------------------------
st.title("Investment Risk Predictor")
st.write("Enter 5 values — the other 13 indicators will be calculated automatically.")

open_p = st.number_input("Open Price", value=100.0)
high_p = st.number_input("High Price", value=101.0)
low_p = st.number_input("Low Price", value=99.0)
close_p = st.number_input("Close Price", value=100.5)
volume_p = st.number_input("Volume", value=1_000_000)

# -----------------------------
# PREDICT
# -----------------------------
if st.button("Predict Risk Level"):

    X = compute_features(open_p, high_p, low_p, close_p, volume_p)
    pred = model.predict(X)[0]

    if pred == 0:
        st.success("🟢 LOW RISK (0)")
    elif pred == 1:
        st.warning("🟡 MEDIUM RISK (1)")
    else:
        st.error("🔴 HIGH RISK (2)")
