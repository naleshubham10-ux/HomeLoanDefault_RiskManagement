"""
Flask web app - Home Loan Default Risk Checker
Run:  python app/app.py   then open http://127.0.0.1:5000
"""
import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import load_model, score  # noqa: E402

app = Flask(__name__)
MODEL, META = load_model()

NUMERIC = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_CHILDREN",
           "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
           "REGION_RATING_CLIENT"]


def form_to_frame(form):
    row = {k: form.get(k) for k in form.keys()}
    for k in NUMERIC:
        v = row.get(k)
        row[k] = float(v) if v not in (None, "") else None
    age = float(row.pop("AGE", 35) or 35)
    years = row.pop("YEARS_EMPLOYED", "")
    row["DAYS_BIRTH"] = -int(age * 365.25)
    row["DAYS_EMPLOYED"] = -int(float(years) * 365.25) if years not in ("", None) else 365243
    row["SK_ID_CURR"] = 0
    return pd.DataFrame([row])


@app.route("/", methods=["GET", "POST"])
def index():
    result, values = None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        res = score(form_to_frame(request.form), MODEL, META).iloc[0]
        result = {"prob": float(res.DEFAULT_PROBABILITY) * 100, "band": res.RISK_BAND,
                  "decision": res.DECISION}
    return render_template("index.html", result=result, values=values, model=META["best_model"],
                           auc=META["metrics"]["ROC_AUC"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    rows = payload if isinstance(payload, list) else [payload]
    return jsonify(score(pd.DataFrame(rows), MODEL, META).to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
