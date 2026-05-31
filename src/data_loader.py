import pandas as pd
from pathlib import Path

def load_csv(path: str = "data/rainfall.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place it under data/ as rainfall.csv")
    df = pd.read_csv(p)
    return df

def preview(path: str = "data/rainfall.csv", n: int = 5):
    df = load_csv(path)
    print(df.head(n))
    print(df.info())