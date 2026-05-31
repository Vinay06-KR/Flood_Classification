# Flood Risk Classification for Farmland

This repository contains a complete Python project to classify flood risk levels for farmland into four classes: No-Risk, Watch, Warning, and Emergency. It uses rainfall and environmental features and treats this as an imbalanced multi-class classification problem.

Project structure

- `data/` - place the Kaggle rainfall dataset here as `rainfall.csv`.
- `src/` - core python modules for data loading, preprocessing, modeling and evaluation.
- `models/` - saved model artifacts.
- `notebooks/` - exploratory analysis notebook.
- `requirements.txt` - Python dependencies.

Quick start

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Download the rainfall dataset from Kaggle and place it at `data/rainfall.csv`.
   Optionally use `scripts/fetch_kaggle_dataset.py` if you have Kaggle credentials configured.

3. Run training example:

```
python -m src.modeling
```

4. Use `src.predict.predict_from_dict()` to predict new samples.

Notes

- If the dataset does not include a `risk` target column, the pipeline will generate heuristic labels from rainfall, soil saturation, and elevation. Inspect `src.preprocess.create_risk_label` to modify rules.
