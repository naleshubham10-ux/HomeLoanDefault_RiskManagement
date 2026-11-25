"""
predict.py
----------
Score loan applicants with the saved model.

Usage:
    python src/predict.py --input new_applicants.csv --output scored.csv
The input must have application_train-style columns (TARGET optional).
Missing bureau / previous-application columns are allowed (treated as NaN).
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import application_features, clean_application  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")


def load_model(model_dir=MODEL_DIR):
    model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    with open(os.path.join(model_dir, "model_meta.json")) as f:
        meta = json.load(f)
    return model, meta


def risk_band(p):
    if p < 0.35:
        return "Low Risk"
    if p < 0.65:
        return "Medium Risk"
    return "High Risk"


def prepare(raw: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = raw.copy()
    # Make sure every raw column used by feature engineering exists
    base = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "AMT_INCOME_TOTAL", "AMT_CREDIT",
            "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "ORGANIZATION_TYPE"]
    for col in base:
        if col not in df:
            df[col] = np.nan
    df["CODE_GENDER"] = df["CODE_GENDER"].fillna("F").replace("XNA", "F")
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH",
                "DAYS_EMPLOYED", "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = application_features(clean_application(df))
    cols = {}
    for col in meta["feature_columns"]:
        if col in df:
            cols[col] = df[col].values
        else:
            cols[col] = [None] * len(df) if meta["feature_dtypes"].get(col) == "object" else np.nan
    X = pd.DataFrame(cols, index=df.index)
    for col, dt in meta["feature_dtypes"].items():
        if dt == "object":
            X[col] = X[col].astype(object)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def score(raw: pd.DataFrame, model=None, meta=None) -> pd.DataFrame:
    if model is None:
        model, meta = load_model()
    X = prepare(raw, meta)
    p = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "SK_ID_CURR": raw.get("SK_ID_CURR", pd.Series(range(len(raw)))).values,
        "DEFAULT_PROBABILITY": np.round(p, 4),
        "PREDICTED_DEFAULT": (p >= meta["threshold"]).astype(int),
        "RISK_BAND": [risk_band(x) for x in p],
        "DECISION": np.where(p >= meta["threshold"], "Manual review / Reject", "Eligible"),
    })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="scored_applicants.csv")
    a = ap.parse_args()
    res = score(pd.read_csv(a.input))
    res.to_csv(a.output, index=False)
    print(res.head(10).to_string(index=False))
    print(f"\nSaved {len(res)} rows -> {a.output}")
