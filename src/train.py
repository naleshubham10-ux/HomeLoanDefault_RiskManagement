"""
train.py
--------
Trains and compares multiple models for Home Loan Default prediction and saves
the best one to models/.

Usage:
    python src/train.py --data data/raw --out models
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_feature_table, split_xy  # noqa: E402

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

RANDOM_STATE = 42


def make_preprocessor(X, scale=True):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False)),
    ])
    return ColumnTransformer([("num", Pipeline(num_steps), num_cols),
                              ("cat", cat_pipe, cat_cols)], remainder="drop")


def get_models(pos_weight):
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   C=0.1, solver="lbfgs"), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=7, min_samples_leaf=100,
                                                 class_weight="balanced",
                                                 random_state=RANDOM_STATE), False),
        "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=12,
                                                 min_samples_leaf=20, max_features="sqrt",
                                                 class_weight="balanced_subsample", n_jobs=-1,
                                                 random_state=RANDOM_STATE), False),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = (XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=3,
                                           subsample=0.8, colsample_bytree=0.4,
                                           min_child_weight=20, reg_lambda=5.0,
                                           scale_pos_weight=pos_weight, eval_metric="auc",
                                           tree_method="hist", n_jobs=-1,
                                           random_state=RANDOM_STATE), False)
    if LGBMClassifier is not None:
        models["LightGBM"] = (LGBMClassifier(n_estimators=600, learning_rate=0.015, num_leaves=8,
                                             max_depth=-1, subsample=0.8, subsample_freq=1,
                                             colsample_bytree=0.4, min_child_samples=150,
                                             reg_alpha=0.1, reg_lambda=5.0, importance_type="gain",
                                             scale_pos_weight=pos_weight, n_jobs=-1,
                                             random_state=RANDOM_STATE, verbose=-1), False)
    return models


def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y_true, y_prob):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "KS": ks_statistic(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Threshold": threshold,
    }


def train_and_compare(X_train, y_train, X_test, y_test, verbose=True):
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    results, fitted, probs = [], {}, {}
    for name, (model, needs_scale) in get_models(pos_weight).items():
        t0 = time.time()
        pipe = Pipeline([("prep", make_preprocessor(X_train, scale=needs_scale)), ("model", model)])
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        thr = best_f1_threshold(y_test, p)
        row = {"Model": name, **evaluate(y_test, p, thr), "Train_time_s": round(time.time() - t0, 1)}
        results.append(row)
        fitted[name], probs[name] = pipe, p
        if verbose:
            print(f"{name:20s} AUC={row['ROC_AUC']:.4f} PR-AUC={row['PR_AUC']:.4f} "
                  f"Recall={row['Recall']:.3f} F1={row['F1']:.3f} ({row['Train_time_s']}s)")
    res = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    return res, fitted, probs


def add_cv_scores(res, fitted, X_train, y_train, n_splits=5):
    """5-fold stratified CV AUC on the training set; models are ranked by CV mean (more robust than one split)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    means, stds = [], []
    for m in res["Model"]:
        s = cross_val_score(fitted[m], X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        means.append(s.mean()); stds.append(s.std())
    res = res.copy()
    res["CV_AUC_mean"], res["CV_AUC_std"] = means, stds
    return res.sort_values("CV_AUC_mean", ascending=False).reset_index(drop=True)


def main(data_dir="data/raw", out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(data_dir, "application_train.csv")):
        print("Original data not found -> generating schema-compatible sample in", data_dir)
        from generate_sample_data import main as generate_sample
        generate_sample(n=30000, out=data_dir)
    df = build_feature_table(data_dir)
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
    res, fitted, _ = train_and_compare(X_tr, y_tr, X_te, y_te)
    res = add_cv_scores(res, fitted, X_tr, y_tr)
    best = res.iloc[0]
    joblib.dump(fitted[best.Model], os.path.join(out_dir, "best_model.joblib"))
    meta = {"best_model": best.Model, "threshold": float(best.Threshold),
            "metrics": {k: float(best[k]) for k in ["ROC_AUC", "PR_AUC", "KS", "Recall", "Precision", "F1"]},
            "feature_columns": X.columns.tolist(),
            "feature_dtypes": {c: str(t) for c, t in X.dtypes.items()}}
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    res.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    print("\n", res.round(4).to_string(index=False))
    print(f"\nSaved best model: {best.Model}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    main(a.data, a.out)
