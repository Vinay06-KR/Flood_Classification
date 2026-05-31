import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # drop exact duplicates
    df = df.drop_duplicates()

    # basic missing handling: numeric -> median, categorical -> mode
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        num_imp = SimpleImputer(strategy="median")
        df[num_cols] = num_imp.fit_transform(df[num_cols])
    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    return df

def create_risk_label(df: pd.DataFrame,
                      rainfall_col: str = "rainfall",
                      soil_col: str = "soil_saturation",
                      elev_col: str = "elevation",
                      target_col: str = "risk") -> pd.DataFrame:
    df = df.copy()
    if target_col in df.columns:
        return df

    # simple heuristic rules to synthesize labels if missing
    r = df.get(rainfall_col)
    s = df.get(soil_col)
    e = df.get(elev_col)

    # ensure numeric
    r = pd.to_numeric(r, errors="coerce").fillna(0)
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    e = pd.to_numeric(e, errors="coerce").fillna(0)

    labels = []
    for rain, soil, elev in zip(r, s, e):
        score = 0
        score += min(rain / 100.0, 1.0) * 2
        score += min(soil / 100.0, 1.0) * 1.5
        if elev < 50:
            score += 1.0

        if score < 1.5:
            labels.append("No-Risk")
        elif score < 2.5:
            labels.append("Watch")
        elif score < 3.5:
            labels.append("Warning")
        else:
            labels.append("Emergency")

    df[target_col] = labels
    return df
