# Detection of Suspicious Customer Activities in E-Commerce Using Machine Learning

MCA Mini Project, Mar Athanasius College of Engineering, Kothamangalam.

An XGBoost model scores e-commerce transactions and sorts each account into
**High**, **Medium** or **Low** risk. A business rule is applied on top of the model:
a purchase made within 1 minute of signup on a device shared by 10 or more accounts
is always marked High risk. The system is served through a Flask web application.

## Files
| File | What it is |
|---|---|
| `fraud_detection_rebuild.ipynb` | Full notebook: EDA, features, split, model training, evaluation |
| `app.py` | Flask web application (routes and pages) |
| `fraud_engine.py` | Feature building, scoring and the business rule used by the web app |
| `templates/`, `static/` | HTML pages and CSS of the web app |
| `fraud_model_xgb_final.json` | Final trained XGBoost model |
| `model_config_final.json` | Feature columns, top countries and risk-tier cut-offs |
| `Fraud_Data.csv`, `IpAddress_to_Country.csv` | Dataset and IP-to-country lookup |
| `New_Review_Batch.csv` | 358 synthetic transactions (2016) for demonstrating batch review on new data |

## Run the web app
```
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000 in a browser.

## Pages
- **Home**: project overview
- **Check Transaction**: score one transaction and see the reasons
- **Batch Review**: upload a CSV of transactions, get a risk-tier summary, review queue and downloadable results

## Results on the unseen test set (22,667 transactions)
ROC-AUC 0.843 · PR-AUC 0.721 · High-tier precision 99.6% · 71.6% of fraud caught in High + Medium.