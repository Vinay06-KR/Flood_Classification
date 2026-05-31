"""Generate a synthetic rainfall dataset, train the model, and save artifact.
This is a demo runner used when the real dataset is not available.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from .preprocess import create_risk_label, clean_data
from .modeling import train_pipeline


def generate_synthetic(n=2000, out_path="data/rainfall.csv"):
    rng = np.random.default_rng(42)
    rainfall = rng.gamma(2.0, 20.0, size=n)  # skewed rainfall
    soil = np.clip(rng.normal(50, 20, size=n), 0, 100)
    elevation = np.clip(rng.normal(100, 80, size=n), 0, 500)
    temperature = np.clip(rng.normal(25, 8, size=n), -10, 50)
    humidity = np.clip(rng.normal(70, 15, size=n), 0, 100)

    df = pd.DataFrame({
        "rainfall": rainfall,
        "soil_saturation": soil,
        "elevation": elevation,
        "temperature": temperature,
        "humidity": humidity,
    })

    df = create_risk_label(df)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def run_demo():
    print("Generating synthetic dataset...")
    df = generate_synthetic()
    print("Dataset generated with shape:", df.shape)

    df = clean_data(df)

    numeric = ["rainfall", "soil_saturation", "elevation", "temperature", "humidity"]
    categorical = []
    features = numeric + categorical

    print("Starting training...")
    model, preproc, test_data = train_pipeline(df, features, numeric, categorical)
    X_test_trans, y_test = test_data
    print("Training complete. Sample test labels:", y_test.unique())


if __name__ == "__main__":
    run_demo()
