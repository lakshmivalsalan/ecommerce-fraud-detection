# E-commerce Fraud Detection

Detects suspicious customer accounts in e-commerce transactions with XGBoost.
Accounts are sorted into High, Medium and Low risk for periodic review.

- `fraud_detection_rebuild.ipynb`: EDA, leakage-free split, features, model comparison, final test
- `App.py`: Streamlit app (batch review, single transaction, about)
- `fraud_model_xgb_final.json`, `model_config_final.json`: final model and settings

Run the app:

    pip install -r requirements.txt
    python -m streamlit run App.py
