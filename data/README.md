# Data

Download the original dataset and extract the 7 CSV files into `data/raw/`:

https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip

Expected files:
- application_train.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- credit_card_balance.csv
- installments_payments.csv

If `data/raw/` is empty, the notebook creates a schema-compatible sample automatically
(`python src/generate_sample_data.py --n 30000 --out data/raw`). Delete `SAMPLE_DATA_FLAG.txt`
together with the sample files when you switch to the original data.
