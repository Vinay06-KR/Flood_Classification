import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def load_model(path: str = "models/xgb_flood.pkl"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train and save a model first.")
    obj = joblib.load(p)
    preprocessor = obj.get("preprocessor")
    model = obj.get("model")
    le = obj.get("label_encoder")
    return preprocessor, model, le

def predict_from_df(df: pd.DataFrame, model_path: str = "models/xgb_flood.pkl"):
    preprocessor, model, le = load_model(model_path)
    X = preprocessor.transform(df)
    probs = model.predict_proba(X)
    preds_enc = model.predict(X)
    if le is not None:
        preds = le.inverse_transform(preds_enc)
    else:
        preds = preds_enc
    return preds, probs

def predict_from_dict(d: dict, model_path: str = "models/xgb_flood.pkl"):
    df = pd.DataFrame([d])
    return predict_from_df(df, model_path)
