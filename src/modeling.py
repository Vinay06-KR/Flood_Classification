from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from .features import build_preprocessor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
import json

def train_pipeline(df: pd.DataFrame,
                   feature_cols,
                   numeric_cols,
                   categorical_cols,
                   target_col: str = "risk",
                   model_path: str = "models/xgb_flood.pkl"):
    X = df[feature_cols]
    y = df[target_col]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # create pipeline: preprocess -> SMOTE -> model. SMOTE applied on numeric array
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # apply preprocessing to training data
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # handle imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_trans, y_train)

    # encode labels
    le = LabelEncoder()
    y_res_enc = le.fit_transform(y_res)

    xgb.fit(X_res, y_res_enc)

    # evaluate on test set (encode y_test)
    try:
        y_test_enc = le.transform(y_test)
    except Exception:
        # if some test labels not seen during training, map them to -1 (won't match predictions)
        y_test_enc = pd.Series(y_test).map(lambda v: -1).values

    y_pred_enc = xgb.predict(X_test_trans)
    # convert preds back to original labels for report
    y_pred = le.inverse_transform(y_pred_enc)
    print(classification_report(y_test, y_pred, zero_division=0))

    # save pipeline components and model + label encoder
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "model": xgb, "label_encoder": le}, model_path)
    print(f"Saved model to {model_path}")
    return xgb, preprocessor, (X_test_trans, y_test)


def tune_model(df: pd.DataFrame,
               feature_cols,
               numeric_cols,
               categorical_cols,
               target_col: str = "risk",
               model_path: str = "models/xgb_flood_tuned.pkl",
               n_iter: int = 8,
               cv=3,
               random_state: int = 42):
    """Quick randomized hyperparameter search using an imblearn pipeline."""
    X = df[feature_cols]
    y = df[target_col]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=0)

    pipe = ImbPipeline([
        ("preproc", preprocessor),
        ("smote", SMOTE(random_state=random_state)),
        ("clf", clf),
    ])

    param_dist = {
        "clf__n_estimators": [50, 100, 150, 200],
        "clf__max_depth": [3, 4, 6, 8],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
    }

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=cv_split, scoring="f1_macro", n_jobs=1, random_state=random_state, verbose=1)
    rs.fit(X, y_enc)

    best = rs.best_estimator_
    # extract components
    best_clf = best.named_steps["clf"]
    best_preproc = best.named_steps["preproc"]

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": best_preproc, "model": best_clf, "label_encoder": le, "search_cv": rs}, model_path)
    print(f"Saved tuned model to {model_path}")
    return rs, best_preproc, best_clf


def save_thresholds(thresholds: dict, path: str = "models/thresholds.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved thresholds to {path}")

if __name__ == "__main__":
    import src.data_loader as dl
    import src.preprocess as pp

    df = dl.load_csv("data/rainfall.csv")
    df = pp.clean_data(df)
    df = pp.create_risk_label(df)

    # naive feature selection - adjust to dataset columns
    possible_numeric = [c for c in df.columns if df[c].dtype.kind in "biufc" and c not in ("id", "risk")]
    numeric = possible_numeric[:4]
    categorical = []
    features = numeric + categorical

    train_pipeline(df, features, numeric, categorical)
