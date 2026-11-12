# FraudShield AI
Premium dark-themed Streamlit case study for credit-card fraud classification.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and upload it in the app sidebar. Required columns: `Time`, `V1`–`V28`, `Amount`, `Class`.

## Deploy
Push this folder to GitHub. On Streamlit Community Cloud select the repository and set the main file path to `app.py`. Keep `requirements.txt` in the repository root.

## Notes
The app trains a class-weight-balanced Logistic Regression model on a stratified train/test split. Metrics and charts are calculated from the uploaded data. This is an educational prototype, not a production fraud-control system. The V1–V28 features are anonymized and predictions are not proof of fraud.
