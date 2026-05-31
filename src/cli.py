"""Simple CLI for training, demo, tuning, optimizing thresholds, and predicting."""
import argparse
from pathlib import Path
import json

from .demo_train import generate_synthetic
from .modeling import train_pipeline, tune_model, save_thresholds
from .preprocess import clean_data
from .predict import predict_from_dict, load_model
from .data_loader import load_csv
from .evaluate import optimize_thresholds_proba


def cmd_demo(args):
    df = generate_synthetic()
    df = clean_data(df)
    numeric = ["rainfall", "soil_saturation", "elevation", "temperature", "humidity"]
    categorical = []
    features = numeric + categorical
    train_pipeline(df, features, numeric, categorical)


def cmd_train(args):
    df = load_csv(args.data)
    df = clean_data(df)
    numeric = [c for c in df.columns if df[c].dtype.kind in "biufc" and c not in ("id", "risk")]
    numeric = numeric[:6]
    categorical = []
    features = numeric + categorical
    train_pipeline(df, features, numeric, categorical, model_path=args.out)


def cmd_tune(args):
    df = load_csv(args.data) if args.data else generate_synthetic()
    df = clean_data(df)
    numeric = ["rainfall", "soil_saturation", "elevation", "temperature", "humidity"]
    categorical = []
    features = numeric + categorical
    rs, preproc, clf = tune_model(df, features, numeric, categorical, n_iter=args.n_iter, model_path=args.out)


def cmd_optimize(args):
    preproc, model, le = load_model(args.model)
    # load dataset
    df = load_csv(args.data) if args.data else None
    if df is None:
        print("Data required for optimization. Provide --data or run demo first.")
        return
    from sklearn.model_selection import train_test_split
    X = df[[c for c in df.columns if c != "risk"]]
    y = df["risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test_trans = preproc.transform(X_test)
    probs = model.predict_proba(X_test_trans)
    classes = list(le.classes_) if le is not None else [str(i) for i in range(probs.shape[1])]
    best = optimize_thresholds_proba(probs, y_test, classes)
    save_thresholds(best)
    print(best)


def cmd_predict(args):
    # parse key=value pairs
    d = dict()
    for kv in args.input:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            val = float(v)
        except Exception:
            val = v
        d[k] = val
    preds, probs = predict_from_dict(d, model_path=args.model)
    print("Prediction:", preds)
    print("Probs:", probs)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd")

    sp_demo = sp.add_parser("demo")
    sp_demo.set_defaults(func=cmd_demo)

    sp_train = sp.add_parser("train")
    sp_train.add_argument("--data", default="data/rainfall.csv")
    sp_train.add_argument("--out", default="models/xgb_flood.pkl")
    sp_train.set_defaults(func=cmd_train)

    sp_tune = sp.add_parser("tune")
    sp_tune.add_argument("--data", default="data/rainfall.csv")
    sp_tune.add_argument("--n_iter", default=6, type=int)
    sp_tune.add_argument("--out", default="models/xgb_flood_tuned.pkl")
    sp_tune.set_defaults(func=cmd_tune)

    sp_opt = sp.add_parser("optimize")
    sp_opt.add_argument("--model", default="models/xgb_flood.pkl")
    sp_opt.add_argument("--data", default="data/rainfall.csv")
    sp_opt.set_defaults(func=cmd_optimize)

    sp_pred = sp.add_parser("predict")
    sp_pred.add_argument("--model", default="models/xgb_flood.pkl")
    sp_pred.add_argument("input", nargs="+", help="key=value pairs for features")
    sp_pred.set_defaults(func=cmd_predict)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
