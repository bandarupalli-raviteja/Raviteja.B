import pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Kidney Stone Prediction", page_icon="🪨", layout="centered")

st.title("🪨 Kidney Stone Prediction (Bi-LSTM)")
st.caption("Enter lab values to estimate kidney stone risk.")

MODEL_PATH = "my_model.h5"
PREPROC_PATH = "preprocessing.pkl"

@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    with open(PREPROC_PATH, "rb") as f:
        pre = pickle.load(f)
    return model, pre

model, pre = load_assets()

mu = pre["mu"]
sigma = pre["sigma"]
q_edges = pre["q_edges"]
feature_cols = pre["feature_cols"]

def to_bins(val, edges):
    idx = np.digitize([val], edges, right=False)[0] - 1
    idx = max(0, min(idx, len(edges)-2))
    return idx

with st.form("inputs"):
    col1, col2 = st.columns(2)
    gravity = col1.number_input("Specific Gravity", value=1.020, step=0.001, format="%.3f")
    ph      = col2.number_input("Urine pH", value=6.00, step=0.01, format="%.2f")
    osmo    = col1.number_input("Osmolality (mOsm/kg)", value=500.0, step=1.0)
    cond    = col2.number_input("Conductivity (mS/cm)", value=20.0, step=0.1)
    urea    = col1.number_input("Urea (mg/dL)", value=300.0, step=1.0)
    calc    = col2.number_input("Calcium (mg/dL)", value=5.0, step=0.01)

    submitted = st.form_submit_button("Predict")

if submitted:
    osmo_cond_ratio = osmo / cond if cond != 0 else 0.0
    urea_calc_diff = urea - calc
    osmo_urea_interaction = osmo * urea

    def z(c, v):
        s = sigma.get(c, 1.0) or 1.0
        return (v - mu.get(c, 0.0)) / s

    g_z = z("gravity", gravity)
    ph_z = z("ph", ph)
    osmo_z = z("osmo", osmo)
    cond_z = z("cond", cond)
    urea_z = z("urea", urea)
    calc_z = z("calc", calc)

    gravity_bin = to_bins(gravity, q_edges["gravity"])
    ph_bin      = to_bins(ph, q_edges["ph"])
    osmo_bin    = to_bins(osmo, q_edges["osmo"])
    cond_bin    = to_bins(cond, q_edges["cond"])
    urea_bin    = to_bins(urea, q_edges["urea"])
    calc_bin    = to_bins(calc, q_edges["calc"])

    x = np.array([[g_z, ph_z, osmo_z, cond_z, urea_z, calc_z,
                   osmo_cond_ratio, urea_calc_diff, osmo_urea_interaction,
                   gravity_bin, ph_bin, osmo_bin, cond_bin, urea_bin, calc_bin]], dtype="float32")
    x = x.reshape((x.shape[0], x.shape[1], 1))

    prob = float(model.predict(x, verbose=0)[0][0])
    st.subheader(f"Risk score: {prob:.3f}")
    if prob >= 0.5:
        st.error("Model suggests **higher** risk of kidney stone.")
    else:
        st.success("Model suggests **lower** risk of kidney stone.")

    with st.expander("Show feature values used by the model"):
        st.write(pd.DataFrame(x.reshape(-1, len(feature_cols)), columns=feature_cols))
