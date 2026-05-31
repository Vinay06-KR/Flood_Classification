import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
import seaborn as sns

def evaluate_model(model, X_test, y_test, class_names=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prf = precision_recall_fscore_support(y_test, y_pred, average=None)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def multiclass_roc_auc(model, X_test, y_test, classes):
    # compute ROC-AUC (one-vs-rest)
    y_score = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=classes)
    try:
        auc = roc_auc_score(y_bin, y_score, average="macro")
        print("ROC-AUC (macro):", auc)
    except Exception:
        print("ROC-AUC not available for this model/data")


def optimize_thresholds_proba(probs, y_true, classes, focus_classes=("Warning", "Emergency")):
    """Simple grid search over thresholds for focus classes to maximize F1 on those classes.

    probs: ndarray shape (n_samples, n_classes)
    y_true: array-like of true labels
    classes: list of class names mapped to probs columns order
    """
    from sklearn.metrics import f1_score
    import numpy as np

    idx_focus = [classes.index(c) for c in focus_classes if c in classes]
    if not idx_focus:
        raise ValueError("No focus classes found in classes list")

    best = {"score": -1, "thresholds": {}}
    grid = np.arange(0.1, 0.91, 0.05)
    # If only one focus class, optimize single threshold
    if len(idx_focus) == 1:
        i = idx_focus[0]
        for t in grid:
            preds = []
            for p in probs:
                if p[i] >= t:
                    preds.append(classes[i])
                else:
                    preds.append(classes[np.argmax(p)])
            score = f1_score(y_true, preds, labels=list(focus_classes), average="macro", zero_division=0)
            if score > best["score"]:
                best["score"] = score
                best["thresholds"] = {classes[i]: float(t)}
    else:
        # two thresholds grid search
        i0, i1 = idx_focus[0], idx_focus[1]
        for t0 in grid:
            for t1 in grid:
                preds = []
                for p in probs:
                    if p[i1] >= t1:
                        preds.append(classes[i1])
                    elif p[i0] >= t0:
                        preds.append(classes[i0])
                    else:
                        preds.append(classes[np.argmax(p)])
                score = f1_score(y_true, preds, labels=list(focus_classes), average="macro", zero_division=0)
                if score > best["score"]:
                    best["score"] = score
                    best["thresholds"] = {classes[i0]: float(t0), classes[i1]: float(t1)}

    return best
