import sys
from pathlib import Path

# ensure project root is on sys.path so `src` imports work when running Streamlit
try:
    ROOT = Path(__file__).resolve().parents[1]
except Exception:
    # fallback if __file__ is not available or parents index fails
    ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from src.predict import predict_from_dict, predict_from_df, load_model
import joblib, json

st.set_page_config(page_title="Flood Risk Classifier", layout="wide")

st.title("Flood Risk Classification — Farmland")
st.write("Predict flood risk levels for farmland using environmental features. Use the form to predict a single sample or upload a CSV for batch predictions.")

MODEL_PATH = "models/xgb_flood_tuned.pkl"
THRESH_PATH = "models/thresholds.json"


@st.cache_resource
def load_resources(model_path=MODEL_PATH, thresh_path=THRESH_PATH):
    preproc, model, le = load_model(model_path)
    thresholds = {}
    if Path(thresh_path).exists():
        with open(thresh_path, "r") as f:
            thresholds = json.load(f)
    # try load example dataset if present for charts
    sample_df = None
    data_path = ROOT / "data" / "rainfall.csv"
    if data_path.exists():
        try:
            sample_df = pd.read_csv(data_path)
        except Exception:
            sample_df = None
    return preproc, model, le, thresholds, sample_df


preproc, model, le, thresholds, sample_df = load_resources()

# Sidebar with model info
with st.sidebar:
    st.header("Model Info")
    st.write("Model:", MODEL_PATH)
    st.write("Thresholds:")
    if thresholds:
        for k, v in thresholds.items():
            st.write(f"- {k}: {v}")
    else:
        st.write("No custom thresholds found.")
    st.markdown("---")
    st.write("Quick example values:")
    st.write({"rainfall":45, "soil_saturation":60, "elevation":30, "temperature":24, "humidity":75})
    st.markdown("---")
    st.write("Risk color legend:")
    st.markdown("""
    <div style='display:flex;gap:8px;align-items:center'>
      <div style='background:#2ecc71;width:18px;height:18px;border-radius:3px'></div><div>No-Risk</div>
      <div style='width:12px'></div>
      <div style='background:#f1c40f;width:18px;height:18px;border-radius:3px'></div><div>Watch</div>
      <div style='width:12px'></div>
      <div style='background:#e67e22;width:18px;height:18px;border-radius:3px'></div><div>Warning</div>
      <div style='width:12px'></div>
      <div style='background:#e74c3c;width:18px;height:18px;border-radius:3px'></div><div>Emergency</div>
    </div>
    """, unsafe_allow_html=True)

tabs = st.tabs(["Predict", "Batch & EDA", "About"])
predict_tab, batch_tab, about_tab = tabs

RISK_COLORS = {
    "No-Risk": "#2ecc71",
    "Watch": "#f1c40f",
    "Warning": "#e67e22",
    "Emergency": "#e74c3c",
}

def render_risk_badge(label: str):
    color = RISK_COLORS.get(label, "#95a5a6")
    html = f"<div style='display:inline-block;padding:6px 12px;background:{color};color:#fff;border-radius:6px;font-weight:600'>{label}</div>"
    st.markdown(html, unsafe_allow_html=True)

with predict_tab:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Single sample prediction")
        with st.form("input_form"):
            rainfall = st.number_input("Rainfall (mm)", value=45.0, step=1.0, format="%.1f")
            soil = st.slider("Soil saturation (%)", min_value=0.0, max_value=100.0, value=60.0)
            elevation = st.number_input("Elevation (m)", value=30.0, step=1.0)
            temperature = st.number_input("Temperature (°C)", value=24.0, step=0.1, format="%.1f")
            humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0)
            submitted = st.form_submit_button("Predict")
    with col2:
        st.subheader("Prediction result")
        result_box = st.empty()
        prob_chart_box = st.empty()

    if submitted:
        sample = {"rainfall": rainfall, "soil_saturation": soil, "elevation": elevation, "temperature": temperature, "humidity": humidity}
        preds, probs = predict_from_dict(sample, model_path=MODEL_PATH)
        pred = preds[0]
        probs_arr = probs[0]
        classes = list(le.classes_) if le is not None else [str(i) for i in range(len(probs_arr))]
        dfp = pd.DataFrame({"class": classes, "prob": probs_arr})
        result_box.success(f"Predicted: {pred}")
        render_risk_badge(pred)
        # show metrics
        top = dfp.sort_values("prob", ascending=False).iloc[0]
        st.metric("Top class", f"{top['class']}", delta=f"{top['prob']:.2%}")
        # probability bar chart
        prob_chart = alt.Chart(dfp).mark_bar().encode(x=alt.X("prob:Q", axis=alt.Axis(format=".0%")), y=alt.Y("class:N", sort="-x"), color=alt.Color("class:N"))
        prob_chart_box.altair_chart(prob_chart, width='stretch')

with batch_tab:
    st.markdown("---")
    st.header("Batch prediction & EDA")
uploaded = st.file_uploader("Upload CSV with columns: rainfall,soil_saturation,elevation,temperature,humidity", type=["csv"]) 
df_up = None
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
else:
    # if sample dataset exists, offer it as demo
    if sample_df is not None:
        if st.checkbox("Use example dataset (data/rainfall.csv) for demo charts and batch predict"):
            df_up = sample_df.copy()

if df_up is not None:
    required = ["rainfall","soil_saturation","elevation","temperature","humidity"]
    if set(required).issubset(df_up.columns):
        preds, probs = predict_from_df(df_up[required], model_path=MODEL_PATH)
        df_up["prediction"] = preds
        st.write(df_up.head())
        st.download_button("Download predictions CSV", df_up.to_csv(index=False), file_name="predictions.csv")
        # show class distribution
        counts = df_up["prediction"].value_counts().reset_index()
        counts.columns = ["class","count"]
        chart = alt.Chart(counts).mark_bar().encode(x="class:N", y="count:Q", color="class:N")
        st.altair_chart(chart, width='stretch')
        # feature histograms
        st.subheader("Feature distributions")
        for col in ["rainfall","soil_saturation","elevation","temperature","humidity"]:
            hist = alt.Chart(df_up).mark_area(opacity=0.3).encode(x=alt.X(f"{col}:Q", bin=True), y='count()')
            st.altair_chart(hist, width='stretch')
        # Risk classification trend (line chart)
        st.subheader("Risk classification trend")
        max_window = max(1, min(100, len(df_up)))
        window = st.slider("Rolling window size (samples)", min_value=1, max_value=max_window, value=min(10, max_window))
        # prefer a date/timestamp column if present
        date_cols = [c for c in df_up.columns if c.lower() in ("date", "timestamp", "time")]
        if date_cols:
            date_col = date_cols[0]
            try:
                df_up[date_col] = pd.to_datetime(df_up[date_col])
                trend = df_up.groupby([date_col, "prediction"]).size().rename("count").reset_index()
                totals = df_up.groupby(date_col).size().rename("total").reset_index()
                trend = trend.merge(totals, on=date_col)
                trend["prop"] = trend["count"] / trend["total"]
                trend_chart = alt.Chart(trend).mark_line(point=True).encode(
                    x=alt.X(f"{date_col}:T", title="Date"),
                    y=alt.Y("prop:Q", axis=alt.Axis(format=".0%")),
                    color=alt.Color("prediction:N", title="Risk class")
                ).interactive()
                st.altair_chart(trend_chart, width='stretch')
            except Exception:
                st.warning("Could not parse date column; using index-based rolling trend instead.")
                # fallthrough to index-based trend
                date_cols = []
        if not date_cols:
            # compute rolling class proportions by index
            classes = df_up["prediction"].unique()
            records = []
            for i in range(len(df_up)):
                start = max(0, i - window + 1)
                win = df_up.iloc[start : i + 1]
                counts = win["prediction"].value_counts(normalize=True)
                for cls in classes:
                    records.append({"idx": i, "class": cls, "prop": counts.get(cls, 0.0)})
            trend_df = pd.DataFrame(records)
            trend_chart = alt.Chart(trend_df).mark_line().encode(
                x=alt.X("idx:Q", title="Sample index (rolling)"),
                y=alt.Y("prop:Q", axis=alt.Axis(format=".0%")),
                color=alt.Color("class:N", title="Risk class")
            ).interactive()
            st.altair_chart(trend_chart, width='stretch')
    else:
        st.error("CSV missing required columns.")

st.markdown("---")
st.caption("Model file: models/xgb_flood_tuned.pkl — thresholds: models/thresholds.json")
