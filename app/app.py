import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.predict import predict_from_dict, predict_from_df, load_model
import joblib, json

st.set_page_config(page_title="Flood Risk Classifier", layout="centered")

st.title("Flood Risk Classification — Farmland")
st.markdown("Provide environmental features to predict flood risk level (No-Risk, Watch, Warning, Emergency).")

MODEL_PATH = "models/xgb_flood_tuned.pkl"
THRESH_PATH = "models/thresholds.json"

@st.cache_resource
def load_resources(model_path=MODEL_PATH, thresh_path=THRESH_PATH):
    preproc, model, le = load_model(model_path)
    thresholds = {}
    if Path(thresh_path).exists():
        with open(thresh_path, "r") as f:
            thresholds = json.load(f)
    return preproc, model, le, thresholds

preproc, model, le, thresholds = load_resources()

with st.form("input_form"):
    rainfall = st.number_input("Rainfall (mm)", value=45.0, step=1.0)
    soil = st.number_input("Soil saturation (%)", value=60.0, min_value=0.0, max_value=100.0)
    elevation = st.number_input("Elevation (m)", value=30.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", value=24.0, step=0.1)
    humidity = st.number_input("Humidity (%)", value=75.0, min_value=0.0, max_value=100.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    sample = {"rainfall": rainfall, "soil_saturation": soil, "elevation": elevation, "temperature": temperature, "humidity": humidity}
    preds, probs = predict_from_dict(sample, model_path=MODEL_PATH)
    st.success(f"Predicted: {preds[0]}")
    st.write("Probabilities:")
    classes = list(le.classes_) if le is not None else [str(i) for i in range(len(probs[0]))]
    dfp = pd.DataFrame([probs[0]], columns=classes)
    st.write(dfp.T)

st.markdown("---")
st.header("Batch prediction")
uploaded = st.file_uploader("Upload CSV with columns: rainfall,soil_saturation,elevation,temperature,humidity", type=["csv"])
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    if set(["rainfall","soil_saturation","elevation","temperature","humidity"]).issubset(df_up.columns):
        preds, probs = predict_from_df(df_up[["rainfall","soil_saturation","elevation","temperature","humidity"]], model_path=MODEL_PATH)
        df_up["prediction"] = preds
        st.write(df_up.head())
        st.download_button("Download predictions CSV", df_up.to_csv(index=False), file_name="predictions.csv")
    else:
        st.error("CSV missing required columns.")

st.markdown("---")
st.caption("Model file: models/xgb_flood_tuned.pkl — thresholds: models/thresholds.json")
