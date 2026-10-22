# Detection of Suspicious Customer Activities in E-Commerce Using Machine Learning

MCA Mini Project, Mar Athanasius College of Engineering, Kothamangalam.

An XGBoost model scores e-commerce transactions and sorts each account into
**High**, **Medium** or **Low** risk. The model is served through **FraudGuard**,
a Flask web application.

## Files
| File | What it is |
|---|---|
| `fraud_detection_rebuild.ipynb` | Full notebook: EDA, features, split, model training, evaluation |
| `app.py` | Flask web application (routes and pages) |
| `fraud_engine.py` | Feature building and scoring used by the web app |
| `templates/`, `static/` | HTML pages and CSS of the web app |
| `fraud_model_xgb_final.json` | Final trained XGBoost model |
| `model_config_final.json` | Feature columns, top countries and risk-tier cut-offs |
| `Fraud_Data.csv`, `IpAddress_to_Country.csv` | Dataset and IP-to-country lookup |

## Run the web app
```
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000 in a browser.

## Pages
- **Home**: project overview and key results
- **Check Transaction**: score one transaction and see the reasons
- **Batch Review**: upload a CSV (or use the demo batch), get a risk-tier summary, review queue and downloadable results
- **Model Dashboard**: model comparison, confusion matrix, risk tiers on unseen data, feature importance

## Results on the unseen test set (22,667 transactions)
ROC-AUC 0.843 · PR-AUC 0.721 · High-tier precision 99.6% · 71.6% of fraud caught in High + Medium.
