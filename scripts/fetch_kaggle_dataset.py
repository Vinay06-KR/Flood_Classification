"""Optional helper to download dataset using Kaggle CLI.
Requires `kaggle` package and configured API token.
"""
from pathlib import Path
import subprocess

def fetch(dataset: str = "agriculture-rainfall-dataset/rainfall", dest: str = "data/rainfall.csv"):
    # dataset string should match Kaggle dataset slug if available
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest_path.parent), "--unzip"]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    try:
        fetch()
        print("Downloaded dataset to data/ (check files)")
    except Exception as e:
        print("Failed to download dataset:", e)