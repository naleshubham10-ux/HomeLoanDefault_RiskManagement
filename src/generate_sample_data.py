"""
generate_sample_data.py
-----------------------
Creates a schema-compatible SAMPLE of the PRCP-1006 Home Credit dataset
(all 7 CSV files, same file names, same key columns, same ID relationships).

Use this ONLY when the original dataset is not available. For the final
submission, download the real data from:
https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip
and extract the CSVs into data/raw/ (they will be picked up automatically).

Usage:
    python src/generate_sample_data.py --n 30000 --out data/raw
"""
import argparse
import os

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_application(n, rng):
    ids = np.arange(100002, 100002 + n)
    gender = rng.choice(["F", "M", "XNA"], n, p=[0.658, 0.3419, 0.0001])
    contract = rng.choice(["Cash loans", "Revolving loans"], n, p=[0.905, 0.095])
    own_car = rng.choice(["N", "Y"], n, p=[0.66, 0.34])
    own_realty = rng.choice(["Y", "N"], n, p=[0.69, 0.31])
    children = rng.choice([0, 1, 2, 3, 4], n, p=[0.70, 0.20, 0.087, 0.012, 0.001])
    income = np.round(np.exp(rng.normal(11.9, 0.5, n)), -2)
    income[rng.random(n) < 0.0005] = 1.17e8 / 100  # a few extreme outliers
    credit = np.round(np.clip(np.exp(rng.normal(13.1, 0.7, n)), 45000, 4.05e6), -2)
    annuity = np.round(credit / rng.uniform(10, 40, n), 1)
    annuity[rng.random(n) < 0.00004] = np.nan
    goods = np.round(credit * rng.uniform(0.8, 1.0, n), -3)
    goods[rng.random(n) < 0.0009] = np.nan

    income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student"],
        n, p=[0.516, 0.233, 0.18, 0.0705, 0.0003, 0.0002])
    education = rng.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher",
         "Lower secondary", "Academic degree"],
        n, p=[0.71, 0.243, 0.033, 0.0135, 0.0005])
    family = rng.choice(["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
                        n, p=[0.639, 0.148, 0.097, 0.064, 0.052])
    housing = rng.choice(["House / apartment", "With parents", "Municipal apartment",
                          "Rented apartment", "Office apartment", "Co-op apartment"],
                         n, p=[0.887, 0.048, 0.036, 0.016, 0.009, 0.004])
    occupation = rng.choice(["Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
                             "High skill tech staff", "Accountants", "Medicine staff",
                             "Security staff", "Cooking staff", None],
                            n, p=[0.18, 0.104, 0.09, 0.07, 0.06, 0.037, 0.032, 0.028,
                                  0.022, 0.0237, 0.3533])
    org = rng.choice(["Business Entity Type 3", "XNA", "Self-employed", "Other", "Medicine",
                      "Government", "School", "Trade: type 7", "Construction", "Transport: type 4"],
                     n, p=[0.25, 0.18, 0.125, 0.06, 0.04, 0.035, 0.03, 0.03, 0.025, 0.225])

    days_birth = -rng.integers(7489, 25229, n)
    days_employed = -rng.exponential(2400, n).astype(int)
    pens = income_type == "Pensioner"
    days_employed[pens & (rng.random(n) < 0.98)] = 365243  # known anomaly in the real data
    days_reg = -rng.integers(0, 24672, n).astype(float)
    days_id = -rng.integers(0, 7197, n)
    days_phone = -rng.integers(0, 4292, n).astype(float)
    region_rating = rng.choice([1, 2, 3], n, p=[0.105, 0.738, 0.157])
    region_pop = np.round(rng.beta(2, 60, n), 6)
    fam_members = children + np.where(np.isin(family, ["Married", "Civil marriage"]), 2, 1)
    own_car_age = np.where(own_car == "Y", rng.integers(0, 40, n), np.nan)

    # External scores (strongest predictors in the real data)
    latent = rng.normal(0, 1, n)
    ext1 = np.clip(_sigmoid(0.9 * latent + rng.normal(0, 0.9, n) + 0.05 * (-days_birth / 3650 - 4.4)), 0.01, 0.96)
    ext2 = np.clip(_sigmoid(0.9 * latent + rng.normal(0.4, 0.8, n)), 0.0001, 0.855)
    ext3 = np.clip(_sigmoid(0.9 * latent + rng.normal(0.1, 0.8, n)), 0.0005, 0.896)
    ext1[rng.random(n) < 0.564] = np.nan
    ext2[rng.random(n) < 0.0021] = np.nan
    ext3[rng.random(n) < 0.198] = np.nan

    # Target generation (~8% default) driven by realistic risk factors
    age_years = -days_birth / 365.25
    emp_years = np.where(days_employed > 0, 0, -days_employed / 365.25)
    logit = (-3.35
             - 1.25 * latent
             - 0.025 * (age_years - 43)
             - 0.04 * np.minimum(emp_years, 20)
             + 0.30 * (region_rating - 2)
             + 0.35 * (gender == "M")
             + 0.25 * (own_car == "N")
             + 0.30 * (education == "Lower secondary")
             - 0.35 * (education == "Higher education")
             + 0.30 * np.isin(income_type, ["Unemployed"]) * 3
             - 0.25 * (income_type == "Pensioner")
             + 0.25 * np.isin(occupation, ["Laborers", "Drivers", "Security staff", "Cooking staff"])
             + 0.20 * (contract == "Cash loans")
             + 0.6 * (annuity / income - 0.18)
             + 0.15 * (family == "Single / not married")
             + rng.normal(0, 0.35, n))
    target = (rng.random(n) < _sigmoid(logit)).astype(int)

    app = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": target, "NAME_CONTRACT_TYPE": contract,
        "CODE_GENDER": gender, "FLAG_OWN_CAR": own_car, "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children, "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods,
        "NAME_TYPE_SUITE": rng.choice(["Unaccompanied", "Family", "Spouse, partner", "Children", None],
                                      n, p=[0.81, 0.13, 0.037, 0.0106, 0.0124]),
        "NAME_INCOME_TYPE": income_type, "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family, "NAME_HOUSING_TYPE": housing,
        "REGION_POPULATION_RELATIVE": region_pop, "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed, "DAYS_REGISTRATION": days_reg,
        "DAYS_ID_PUBLISH": days_id, "OWN_CAR_AGE": own_car_age,
        "FLAG_MOBIL": 1, "FLAG_EMP_PHONE": (~pens).astype(int),
        "FLAG_WORK_PHONE": rng.choice([0, 1], n, p=[0.8, 0.2]),
        "FLAG_PHONE": rng.choice([0, 1], n, p=[0.72, 0.28]),
        "FLAG_EMAIL": rng.choice([0, 1], n, p=[0.943, 0.057]),
        "OCCUPATION_TYPE": occupation, "CNT_FAM_MEMBERS": fam_members.astype(float),
        "REGION_RATING_CLIENT": region_rating,
        "REGION_RATING_CLIENT_W_CITY": np.clip(region_rating + rng.choice([-1, 0, 0, 0, 0, 1], n), 1, 3),
        "WEEKDAY_APPR_PROCESS_START": rng.choice(["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
                                                  "FRIDAY", "SATURDAY", "SUNDAY"], n),
        "HOUR_APPR_PROCESS_START": rng.integers(0, 24, n),
        "REG_CITY_NOT_LIVE_CITY": rng.choice([0, 1], n, p=[0.92, 0.08]),
        "REG_CITY_NOT_WORK_CITY": rng.choice([0, 1], n, p=[0.77, 0.23]),
        "ORGANIZATION_TYPE": org,
        "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        "APARTMENTS_AVG": np.where(rng.random(n) < 0.507, np.nan, np.round(rng.beta(2, 15, n), 4)),
        "TOTALAREA_MODE": np.where(rng.random(n) < 0.483, np.nan, np.round(rng.beta(2, 18, n), 4)),
        "OBS_30_CNT_SOCIAL_CIRCLE": rng.poisson(1.4, n).astype(float),
        "DEF_30_CNT_SOCIAL_CIRCLE": (rng.poisson(0.14, n) + (target & (rng.random(n) < 0.08))).astype(float),
        "DAYS_LAST_PHONE_CHANGE": days_phone,
        "FLAG_DOCUMENT_3": rng.choice([0, 1], n, p=[0.29, 0.71]),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(n) < 0.135, np.nan, rng.poisson(1.9, n)),
    })
    return app, latent


def make_bureau(app, latent, rng):
    n = len(app)
    counts = rng.poisson(5.0, n)
    counts[rng.random(n) < 0.14] = 0
    cur = np.repeat(app.SK_ID_CURR.values, counts)
    lat = np.repeat(latent, counts)
    m = len(cur)
    active = rng.choice(["Closed", "Active", "Sold", "Bad debt"], m, p=[0.628, 0.367, 0.0049, 0.0001])
    credit_sum = np.round(np.exp(rng.normal(11.8, 1.1, m)), 1)
    debt = np.where(active == "Active", credit_sum * rng.uniform(0, 1, m), 0)
    overdue = np.where(rng.random(m) < _sigmoid(-3.8 - 0.8 * lat), rng.exponential(5000, m), 0)
    return pd.DataFrame({
        "SK_ID_CURR": cur, "SK_ID_BUREAU": np.arange(5000000, 5000000 + m),
        "CREDIT_ACTIVE": active, "CREDIT_CURRENCY": "currency 1",
        "DAYS_CREDIT": -rng.integers(0, 2922, m),
        "CREDIT_DAY_OVERDUE": np.where(overdue > 0, rng.integers(1, 120, m), 0),
        "DAYS_CREDIT_ENDDATE": rng.integers(-2900, 3000, m).astype(float),
        "AMT_CREDIT_MAX_OVERDUE": np.where(rng.random(m) < 0.65, np.nan, overdue),
        "CNT_CREDIT_PROLONG": rng.choice([0, 1, 2], m, p=[0.994, 0.005, 0.001]),
        "AMT_CREDIT_SUM": credit_sum,
        "AMT_CREDIT_SUM_DEBT": np.where(rng.random(m) < 0.15, np.nan, np.round(debt, 1)),
        "AMT_CREDIT_SUM_OVERDUE": np.round(overdue, 1),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan", "Mortgage",
                                   "Microloan"], m, p=[0.73, 0.234, 0.017, 0.011, 0.008]),
        "DAYS_CREDIT_UPDATE": -rng.integers(0, 2900, m),
    })


def make_bureau_balance(bureau, rng, max_bureau=60000):
    b = bureau.sample(min(len(bureau), max_bureau), random_state=1)
    months = rng.integers(1, 20, len(b))
    bid = np.repeat(b.SK_ID_BUREAU.values, months)
    mb = np.concatenate([-np.arange(k) for k in months])
    status = rng.choice(["C", "0", "X", "1", "2", "3", "4", "5"], len(bid),
                        p=[0.5, 0.27, 0.2, 0.022, 0.004, 0.0015, 0.001, 0.0015])
    return pd.DataFrame({"SK_ID_BUREAU": bid, "MONTHS_BALANCE": mb, "STATUS": status})


def make_previous(app, latent, rng):
    n = len(app)
    counts = rng.poisson(4.5, n)
    counts[rng.random(n) < 0.05] = 0
    cur = np.repeat(app.SK_ID_CURR.values, counts)
    lat = np.repeat(latent, counts)
    m = len(cur)
    status = np.where(rng.random(m) < _sigmoid(-1.3 - 0.6 * lat), "Refused",
                      rng.choice(["Approved", "Canceled", "Unused offer"], m, p=[0.8, 0.18, 0.02]))
    app_amt = np.round(np.exp(rng.normal(11.2, 1.2, m)), 1)
    credit_amt = np.where(status == "Approved", app_amt * rng.uniform(0.9, 1.2, m), 0)
    return pd.DataFrame({
        "SK_ID_PREV": np.arange(1000000, 1000000 + m), "SK_ID_CURR": cur,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Consumer loans", "Revolving loans"],
                                         m, p=[0.45, 0.44, 0.11]),
        "AMT_ANNUITY": np.round(app_amt / rng.uniform(6, 36, m), 1),
        "AMT_APPLICATION": app_amt, "AMT_CREDIT": np.round(credit_amt, 1),
        "AMT_DOWN_PAYMENT": np.where(rng.random(m) < 0.54, np.nan, np.round(app_amt * rng.uniform(0, 0.2, m), 1)),
        "AMT_GOODS_PRICE": np.where(rng.random(m) < 0.23, np.nan, app_amt),
        "NAME_CONTRACT_STATUS": status,
        "DAYS_DECISION": -rng.integers(1, 2922, m),
        "NAME_CLIENT_TYPE": rng.choice(["Repeater", "New", "Refreshed"], m, p=[0.74, 0.18, 0.08]),
        "CNT_PAYMENT": np.where(status == "Approved", rng.choice([6, 10, 12, 18, 24, 36, 48], m), np.nan),
        "NAME_YIELD_GROUP": rng.choice(["XNA", "middle", "high", "low_normal", "low_action"],
                                       m, p=[0.31, 0.23, 0.21, 0.19, 0.06]),
        "DAYS_FIRST_DUE": np.where(status == "Approved", -rng.integers(1, 2900, m), np.nan),
    })


def make_installments(prev, latent_map, rng, max_prev=70000):
    p = prev[prev.NAME_CONTRACT_STATUS == "Approved"].sample(
        min(max_prev, (prev.NAME_CONTRACT_STATUS == "Approved").sum()), random_state=2)
    k = rng.integers(3, 15, len(p))
    sid = np.repeat(p.SK_ID_PREV.values, k)
    cur = np.repeat(p.SK_ID_CURR.values, k)
    lat = latent_map.reindex(cur).values
    inst = np.repeat(p.AMT_ANNUITY.values, k)
    m = len(sid)
    num = np.concatenate([np.arange(1, x + 1) for x in k])
    due = np.repeat(p.DAYS_FIRST_DUE.fillna(-500).values, k) + 30 * (num - 1)
    late = np.where(rng.random(m) < _sigmoid(-2.3 - 0.9 * lat), rng.integers(1, 60, m), -rng.integers(0, 15, m))
    under = rng.random(m) < _sigmoid(-3.0 - 0.7 * lat)
    return pd.DataFrame({
        "SK_ID_PREV": sid, "SK_ID_CURR": cur, "NUM_INSTALMENT_VERSION": 1.0,
        "NUM_INSTALMENT_NUMBER": num, "DAYS_INSTALMENT": due.astype(float),
        "DAYS_ENTRY_PAYMENT": (due + late).astype(float), "AMT_INSTALMENT": inst,
        "AMT_PAYMENT": np.round(np.where(under, inst * rng.uniform(0.1, 0.9, m), inst), 2),
    })


def make_pos(prev, latent_map, rng, max_prev=50000):
    p = prev[prev.NAME_CONTRACT_TYPE != "Revolving loans"].sample(
        min(max_prev, (prev.NAME_CONTRACT_TYPE != "Revolving loans").sum()), random_state=3)
    k = rng.integers(2, 12, len(p))
    sid = np.repeat(p.SK_ID_PREV.values, k)
    cur = np.repeat(p.SK_ID_CURR.values, k)
    lat = latent_map.reindex(cur).values
    m = len(sid)
    mb = np.concatenate([-np.arange(1, x + 1) for x in k])
    dpd = np.where(rng.random(m) < _sigmoid(-3.2 - 0.8 * lat), rng.integers(1, 90, m), 0)
    return pd.DataFrame({
        "SK_ID_PREV": sid, "SK_ID_CURR": cur, "MONTHS_BALANCE": mb,
        "CNT_INSTALMENT": rng.choice([6, 12, 24, 36], m).astype(float),
        "CNT_INSTALMENT_FUTURE": rng.integers(0, 24, m).astype(float),
        "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed", "Signed"], m, p=[0.91, 0.07, 0.02]),
        "SK_DPD": dpd, "SK_DPD_DEF": np.where(dpd > 30, dpd - 30, 0),
    })


def make_credit_card(app, latent, rng, frac=0.28):
    sel = rng.random(len(app)) < frac
    cur = app.SK_ID_CURR.values[sel]
    lat_c = latent[sel]
    k = rng.integers(3, 18, len(cur))
    cid = np.repeat(cur, k)
    lat = np.repeat(lat_c, k)
    sid = np.repeat(np.arange(2000000, 2000000 + len(cur)), k)
    m = len(cid)
    limit = np.repeat(rng.choice([45000, 90000, 135000, 180000, 270000], len(cur)), k).astype(float)
    util = np.clip(_sigmoid(-0.3 - 0.9 * lat + rng.normal(0, 0.8, m)), 0, 1.1)
    bal = np.round(limit * util, 2)
    dpd = np.where(rng.random(m) < _sigmoid(-3.5 - 0.9 * lat), rng.integers(1, 60, m), 0)
    return pd.DataFrame({
        "SK_ID_PREV": sid, "SK_ID_CURR": cid,
        "MONTHS_BALANCE": np.concatenate([-np.arange(1, x + 1) for x in k]),
        "AMT_BALANCE": bal, "AMT_CREDIT_LIMIT_ACTUAL": limit,
        "AMT_DRAWINGS_CURRENT": np.round(bal * rng.uniform(0, 0.3, m), 2),
        "AMT_PAYMENT_CURRENT": np.where(rng.random(m) < 0.2, np.nan, np.round(bal * rng.uniform(0.02, 0.3, m), 2)),
        "AMT_INST_MIN_REGULARITY": np.round(bal * 0.05, 2),
        "CNT_DRAWINGS_CURRENT": rng.poisson(1.5, m),
        "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed", "Signed"], m, p=[0.96, 0.03, 0.01]),
        "SK_DPD": dpd, "SK_DPD_DEF": np.where(dpd > 30, dpd - 30, 0),
    })


def main(n=30000, out="data/raw", seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    app, latent = make_application(n, rng)
    latent_map = pd.Series(latent, index=app.SK_ID_CURR.values)
    bureau = make_bureau(app, latent, rng)
    tables = {
        "application_train.csv": app,
        "bureau.csv": bureau,
        "bureau_balance.csv": make_bureau_balance(bureau, rng),
    }
    prev = make_previous(app, latent, rng)
    tables["previous_application.csv"] = prev
    tables["installments_payments.csv"] = make_installments(prev, latent_map, rng)
    tables["POS_CASH_balance.csv"] = make_pos(prev, latent_map, rng)
    tables["credit_card_balance.csv"] = make_credit_card(app, latent, rng)
    for name, df in tables.items():
        df.to_csv(os.path.join(out, name), index=False)
        print(f"{name:28s} {df.shape}")
    with open(os.path.join(out, "SAMPLE_DATA_FLAG.txt"), "w") as f:
        f.write("These CSVs were generated by src/generate_sample_data.py (schema-compatible sample).\n"
                "Replace them with the original PRCP-1006 files for final results.\n")
    print(f"Default rate: {app.TARGET.mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.n, a.out, a.seed)
