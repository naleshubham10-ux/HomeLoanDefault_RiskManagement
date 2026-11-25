# PRCP-1006 · Home Loan Default – Risk Management

Predict which home-loan applicants are likely to **default**, and identify the factors and customer segments that are **eligible for a loan**.
Domain: Banking · Data: Home Credit (7 linked tables).

## Deliverables (all in one notebook)
| Task | Where |
|---|---|
| Task 1 – Complete data analysis report | `notebooks/PRCP-1006-HomeLoanDef.ipynb` – Section 2 |
| Task 2 – Predictive model, factors & eligible segments | Sections 3, 4, 6 |
| Model comparison report + best model for production | Section 5 |
| Report on challenges faced (technique + reason) | Section 7 |
| Project report (Word) | `reports/PRCP-1006_HomeLoanDefault_Project_Report.docx` |
| Presentation (PowerPoint) | `reports/PRCP-1006_HomeLoanDefault_Presentation.pptx` |

## Results (included run)
| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | KS |
|---|---|---|---|---|
| **LightGBM (selected)** | **0.783** | **0.773** | **0.257** | **0.416** |
| XGBoost | 0.783 | 0.773 | 0.257 | 0.416 |
| Logistic Regression | 0.780 | 0.765 | 0.250 | 0.401 |
| Random Forest | 0.775 | 0.763 | 0.246 | 0.404 |
| Decision Tree | 0.727 | 0.717 | 0.195 | 0.355 |

> These numbers come from a schema-compatible sample (30,000 applicants) built by `src/generate_sample_data.py`.
> Put the original PRCP-1006 CSVs in `data/raw/` and re-run the notebook to get final numbers on the full data.

## Folder structure
```
HomeLoanDefault_RiskManagement/
├── notebooks/PRCP-1006-HomeLoanDef.ipynb   # single submission notebook (executed, with outputs)
├── src/
│   ├── features.py               # cleaning + feature engineering for all 7 tables
│   ├── train.py                  # trains 5 models, CV, saves best model
│   ├── predict.py                # batch scoring of new applicants
│   └── generate_sample_data.py   # schema-compatible sample generator
├── app/                          # Flask web app + REST API
│   ├── app.py
│   └── templates/index.html
├── models/                       # best_model.joblib, model_meta.json, result CSVs
├── reports/                      # Word report, PPT, figures/
├── data/raw/                     # put the original CSVs here
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt

# 1. Data: download and extract into data/raw/
#    https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip
#    (if data/raw is empty, the notebook generates a sample automatically)

# 2. Notebook (all tasks)
jupyter notebook notebooks/PRCP-1006-HomeLoanDef.ipynb

# 3. Or train from the command line
python src/train.py --data data/raw --out models

# 4. Score new applicants
python src/predict.py --input new_applicants.csv --output scored.csv

# 5. Web app  ->  http://127.0.0.1:5000
python app/app.py
```

### REST API example
```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
  -d '{"CODE_GENDER":"F","DAYS_BIRTH":-18000,"DAYS_EMPLOYED":-3000,"AMT_INCOME_TOTAL":250000,
       "AMT_CREDIT":500000,"AMT_ANNUITY":25000,"EXT_SOURCE_2":0.75,"EXT_SOURCE_3":0.7}'
```
Returns `DEFAULT_PROBABILITY`, `PREDICTED_DEFAULT`, `RISK_BAND` (Low / Medium / High) and `DECISION`.

## Notes on the full dataset
- The full data is ~2.5 GB; 8 GB+ RAM is recommended. `features.py` down-casts types and aggregates each table before merging.
- On the full data, expect the notebook to take roughly 15–30 minutes (Random Forest and 5-fold CV are the slowest steps).

## Author
Shubham Nale · B.Tech Computer Science & Engineering · Arvind Gavali College of Engineering, Satara (DBATU)
