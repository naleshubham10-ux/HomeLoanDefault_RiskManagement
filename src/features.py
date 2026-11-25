"""
features.py
-----------
Data loading, cleaning and feature engineering for PRCP-1006 Home Loan Default.
Works with the ORIGINAL Home Credit CSVs and with the schema-compatible sample.

Main entry point:
    df = build_feature_table(data_dir)   # one row per SK_ID_CURR, with TARGET
"""
import gc
import os

import numpy as np
import pandas as pd

DAYS_EMPLOYED_ANOMALY = 365243


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def reduce_memory(df):
    """Downcast numeric columns to save RAM (the real dataset is ~2.5 GB)."""
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def read_csv(data_dir, name, nrows=None):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        print(f"[warn] {name} not found - skipping")
        return None
    return reduce_memory(pd.read_csv(path, nrows=nrows))


def one_hot(df, nan_as_category=True):
    cats = [c for c in df.columns if df[c].dtype == "object"]
    df = pd.get_dummies(df, columns=cats, dummy_na=nan_as_category, dtype=np.uint8)
    new_cols = [c for c in df.columns if any(c.startswith(k + "_") for k in cats)]
    return df, new_cols


def _agg(df, group_key, agg_map, prefix):
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}
    out = df.groupby(group_key).agg(agg_map)
    out.columns = [f"{prefix}_{a}_{b.upper()}" for a, b in out.columns]
    return out


def safe_div(a, b):
    return a / b.replace(0, np.nan)


# ----------------------------------------------------------------------------
# application_train
# ----------------------------------------------------------------------------
def clean_application(app):
    app = app.copy()
    # 1. Gender 'XNA' -> very rare, drop rows (4 rows in the real data)
    app = app[app["CODE_GENDER"] != "XNA"]
    # 2. DAYS_EMPLOYED anomaly 365243 (~18% of rows, mostly pensioners)
    app["DAYS_EMPLOYED_ANOM"] = (app["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype(np.uint8)
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)
    # 3. XNA organisation -> NaN
    if "ORGANIZATION_TYPE" in app:
        app["ORGANIZATION_TYPE"] = app["ORGANIZATION_TYPE"].replace("XNA", np.nan)
    # 4. Cap extreme income outliers at 99.9 percentile
    cap = app["AMT_INCOME_TOTAL"].quantile(0.999)
    app["AMT_INCOME_TOTAL"] = app["AMT_INCOME_TOTAL"].clip(upper=cap)
    return app


def application_features(app):
    app = app.copy()
    app["AGE_YEARS"] = -app["DAYS_BIRTH"] / 365.25
    app["EMPLOYED_YEARS"] = -app["DAYS_EMPLOYED"] / 365.25
    app["CREDIT_INCOME_RATIO"] = safe_div(app["AMT_CREDIT"], app["AMT_INCOME_TOTAL"])
    app["ANNUITY_INCOME_RATIO"] = safe_div(app["AMT_ANNUITY"], app["AMT_INCOME_TOTAL"])
    app["CREDIT_TERM"] = safe_div(app["AMT_ANNUITY"], app["AMT_CREDIT"])
    app["GOODS_CREDIT_RATIO"] = safe_div(app["AMT_GOODS_PRICE"], app["AMT_CREDIT"])
    app["EMPLOYED_AGE_RATIO"] = safe_div(app["DAYS_EMPLOYED"], app["DAYS_BIRTH"])
    app["INCOME_PER_PERSON"] = safe_div(app["AMT_INCOME_TOTAL"], app["CNT_FAM_MEMBERS"])
    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in app]
    app["EXT_SOURCE_MEAN"] = app[ext].mean(axis=1)
    app["EXT_SOURCE_MIN"] = app[ext].min(axis=1)
    app["EXT_SOURCE_MAX"] = app[ext].max(axis=1)
    app["EXT_SOURCE_STD"] = app[ext].std(axis=1)
    app["EXT_SOURCE_PROD"] = app[ext].prod(axis=1, min_count=1)
    app["EXT_SOURCE_NAN_COUNT"] = app[ext].isna().sum(axis=1)
    doc_cols = [c for c in app.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        app["DOCUMENT_COUNT"] = app[doc_cols].sum(axis=1)
    # Binary encodings
    for col, pos in [("CODE_GENDER", "M"), ("FLAG_OWN_CAR", "Y"), ("FLAG_OWN_REALTY", "Y")]:
        if col in app:
            app[col] = (app[col] == pos).astype(np.uint8)
    return app


# ----------------------------------------------------------------------------
# bureau + bureau_balance
# ----------------------------------------------------------------------------
def bureau_features(bureau, bb):
    if bureau is None:
        return None
    if bb is not None:
        bb = bb.copy()
        bb["STATUS_DPD"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(np.uint8)
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(
            BB_MONTHS=("MONTHS_BALANCE", "size"), BB_DPD_RATE=("STATUS_DPD", "mean"))
        bureau = bureau.merge(bb_agg, left_on="SK_ID_BUREAU", right_index=True, how="left")
        del bb, bb_agg
        gc.collect()

    bureau = bureau.copy()
    bureau["DEBT_CREDIT_RATIO"] = safe_div(bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"])
    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(np.uint8)
    bureau["IS_BAD"] = bureau["CREDIT_ACTIVE"].isin(["Bad debt", "Sold"]).astype(np.uint8)
    bureau["HAS_OVERDUE"] = (bureau["CREDIT_DAY_OVERDUE"] > 0).astype(np.uint8)
    num = {
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "DEBT_CREDIT_RATIO": ["mean", "max"],
        "IS_ACTIVE": ["sum", "mean"],
        "IS_BAD": ["sum"],
        "HAS_OVERDUE": ["sum", "mean"],
        "BB_MONTHS": ["mean", "sum"],
        "BB_DPD_RATE": ["mean", "max"],
        "SK_ID_BUREAU": ["count"],
    }
    agg = _agg(bureau, "SK_ID_CURR", num, "BUREAU")
    # Credit-type mix
    if "CREDIT_TYPE" in bureau:
        ct = pd.crosstab(bureau["SK_ID_CURR"], bureau["CREDIT_TYPE"], normalize="index")
        keep = [c for c in ["Consumer credit", "Credit card", "Car loan", "Mortgage", "Microloan"] if c in ct]
        ct = ct[keep]
        ct.columns = ["BUREAU_TYPE_" + c.upper().replace(" ", "_") for c in ct.columns]
        agg = agg.join(ct)
    return agg


# ----------------------------------------------------------------------------
# previous_application
# ----------------------------------------------------------------------------
def previous_features(prev):
    if prev is None:
        return None
    prev = prev.copy()
    for c in ["DAYS_FIRST_DUE", "DAYS_DECISION"]:
        if c in prev:
            prev[c] = prev[c].replace(DAYS_EMPLOYED_ANOMALY, np.nan)
    prev["APP_CREDIT_RATIO"] = safe_div(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.uint8)
    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.uint8)
    num = {
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max", "sum"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "APP_CREDIT_RATIO": ["mean", "max"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
        "IS_REFUSED": ["sum", "mean"],
        "IS_APPROVED": ["sum", "mean"],
        "SK_ID_PREV": ["count"],
    }
    return _agg(prev, "SK_ID_CURR", num, "PREV")


# ----------------------------------------------------------------------------
# POS_CASH_balance / installments / credit_card_balance
# ----------------------------------------------------------------------------
def pos_features(pos):
    if pos is None:
        return None
    pos = pos.copy()
    pos["DPD_FLAG"] = (pos["SK_DPD"] > 0).astype(np.uint8)
    num = {"MONTHS_BALANCE": ["min", "size"], "SK_DPD": ["max", "mean"],
           "SK_DPD_DEF": ["max", "mean"], "DPD_FLAG": ["mean"],
           "CNT_INSTALMENT_FUTURE": ["mean"]}
    return _agg(pos, "SK_ID_CURR", num, "POS")


def installments_features(ins):
    if ins is None:
        return None
    ins = ins.copy()
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    ins["LATE"] = (ins["DPD"] > 0).astype(np.uint8)
    ins["PAYMENT_PERC"] = safe_div(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["UNDERPAID"] = (ins["PAYMENT_DIFF"] > 1).astype(np.uint8)
    num = {"DPD": ["max", "mean", "sum"], "DBD": ["max", "mean"], "LATE": ["mean", "sum"],
           "PAYMENT_PERC": ["mean", "min"], "PAYMENT_DIFF": ["mean", "max", "sum"],
           "UNDERPAID": ["mean"], "AMT_PAYMENT": ["sum", "mean"], "SK_ID_PREV": ["count"]}
    return _agg(ins, "SK_ID_CURR", num, "INSTAL")


def credit_card_features(cc):
    if cc is None:
        return None
    cc = cc.copy()
    cc["UTILIZATION"] = safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["DPD_FLAG"] = (cc["SK_DPD"] > 0).astype(np.uint8)
    num = {"AMT_BALANCE": ["mean", "max"], "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
           "UTILIZATION": ["mean", "max"], "AMT_DRAWINGS_CURRENT": ["mean", "sum"],
           "AMT_PAYMENT_CURRENT": ["mean"], "CNT_DRAWINGS_CURRENT": ["mean", "sum"],
           "SK_DPD": ["max", "mean"], "DPD_FLAG": ["mean"], "MONTHS_BALANCE": ["size"]}
    return _agg(cc, "SK_ID_CURR", num, "CC")


# ----------------------------------------------------------------------------
# Master table
# ----------------------------------------------------------------------------
def build_feature_table(data_dir, nrows=None, verbose=True):
    app = read_csv(data_dir, "application_train.csv", nrows)
    app = application_features(clean_application(app))

    blocks = {
        "bureau": lambda: bureau_features(read_csv(data_dir, "bureau.csv"),
                                          read_csv(data_dir, "bureau_balance.csv")),
        "previous": lambda: previous_features(read_csv(data_dir, "previous_application.csv")),
        "pos": lambda: pos_features(read_csv(data_dir, "POS_CASH_balance.csv")),
        "installments": lambda: installments_features(read_csv(data_dir, "installments_payments.csv")),
        "credit_card": lambda: credit_card_features(read_csv(data_dir, "credit_card_balance.csv")),
    }
    for name, fn in blocks.items():
        agg = fn()
        if agg is not None:
            app = app.merge(agg, left_on="SK_ID_CURR", right_index=True, how="left")
            if verbose:
                print(f"merged {name:13s} -> +{agg.shape[1]:3d} features | shape {app.shape}")
        gc.collect()

    # Flags for "no history" in each source (missing after the left join)
    for pfx, col in [("BUREAU", "BUREAU_SK_ID_BUREAU_COUNT"), ("PREV", "PREV_SK_ID_PREV_COUNT"),
                     ("INSTAL", "INSTAL_SK_ID_PREV_COUNT"), ("CC", "CC_MONTHS_BALANCE_SIZE")]:
        if col in app:
            app[f"{pfx}_NO_HISTORY"] = app[col].isna().astype(np.uint8)

    app = app.replace([np.inf, -np.inf], np.nan)
    return app


def split_xy(df):
    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    return X, y
