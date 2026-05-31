from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def select_features(df: pd.DataFrame, feature_cols: List[str], target_col: str = "risk") -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]):
    num_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])
    return preprocessor
