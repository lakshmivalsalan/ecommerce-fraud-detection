"""
FraudGuard: Flask web application for the mini project
"Detection of Suspicious Customer Activities in E-Commerce Using Machine Learning".

Run with:   python app.py      then open  http://127.0.0.1:5000
"""
import io
import uuid

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, flash

from fraud_engine import FraudEngine, ip_to_number, TIERS, TIER_ACTION, TEST_TIERS

app = Flask(__name__)
app.secret_key = "fraudguard-mini-project"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

engine = FraudEngine()
RESULTS = {}          # last scored batches, kept in memory for download


def tier_summary(scored):
    total = len(scored)
    rows = []
    for t in TIERS:
        part = scored[scored["risk_tier"] == t]
        row = {"tier": t, "count": len(part), "share": 100 * len(part) / total if total else 0,
               "action": TIER_ACTION[t]}
        if "actual_fraud" in scored.columns:
            frauds = int(part["actual_fraud"].sum())
            row["frauds"] = frauds
            row["fraud_rate"] = 100 * frauds / len(part) if len(part) else 0
        rows.append(row)
    return rows


@app.route("/")
def home():
    return render_template("index.html", active="home")


@app.route("/check", methods=["GET", "POST"])
def check():
    form = {"value": 40, "age": 33, "sex": "M", "browser": "Chrome", "source": "SEO",
            "ip": "66.249.66.1", "signup": "2015-05-01 09:30:00",
            "purchase": "2015-06-12 18:45:00", "devices": 1}
    result, error = None, None
    if request.method == "POST":
        form.update({k: request.form.get(k, v) for k, v in form.items()})
        try:
            value = float(form["value"]); age = int(form["age"]); devices = int(form["devices"])
        except ValueError:
            error = "Purchase value, age and device count must be numbers."
        else:
            ip_num = ip_to_number(form["ip"])
            signup = pd.to_datetime(form["signup"], errors="coerce")
            purchase = pd.to_datetime(form["purchase"], errors="coerce")
            if ip_num is None:
                error = "The IP address should look like 66.249.66.1."
            elif pd.isna(signup) or pd.isna(purchase):
                error = "Dates should look like 2015-05-01 09:30:00."
            elif purchase < signup:
                error = "The purchase time is before the signup time."
            elif devices < 1:
                error = "The device count must be at least 1."
            else:
                result = engine.score_one(signup, purchase, value, age, form["sex"],
                                          form["browser"], form["source"], ip_num, devices)
    return render_template("check.html", active="check", form=form, result=result,
                           error=error, cfg=engine.config)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", active="batch", scored=None)

    try:
        if request.form.get("demo"):
            raw = engine.load_demo()
            if raw is None:
                flash("Fraud_Data.csv was not found, so the demo batch is not available.")
                return redirect(url_for("batch"))
            source_name = "Demo batch from Fraud_Data.csv"
        else:
            file = request.files.get("file")
            if not file or not file.filename.lower().endswith(".csv"):
                flash("Please choose a CSV file.")
                return redirect(url_for("batch"))
            raw = pd.read_csv(file)
            source_name = file.filename
        scored = engine.score_batch(raw)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("batch"))

    token = uuid.uuid4().hex
    RESULTS.clear()
    RESULTS[token] = scored
    queue = (scored[scored["risk_tier"] != "Low"]
             .sort_values("risk_score", ascending=False).head(100))
    return render_template("batch.html", active="batch", scored=scored, token=token,
                           summary=tier_summary(scored), queue=queue.to_dict("records"),
                           source_name=source_name, has_label="actual_fraud" in scored.columns)


@app.route("/download/<token>")
def download(token):
    scored = RESULTS.get(token)
    if scored is None:
        flash("That result has expired. Please score the batch again.")
        return redirect(url_for("batch"))
    buf = io.BytesIO(scored.to_csv(index=False).encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="scored_transactions.csv")


@app.route("/dashboard")
def dashboard():
    total = sum(v[0] for v in TEST_TIERS.values())
    tiers = [{"tier": t, "count": TEST_TIERS[t][0], "frauds": TEST_TIERS[t][1],
              "share": 100 * TEST_TIERS[t][0] / total,
              "rate": 100 * TEST_TIERS[t][1] / TEST_TIERS[t][0], "action": TIER_ACTION[t]}
             for t in TIERS]
    models = [
        ("Logistic Regression", 0.955, 0.996, 0.524, 0.686, 0.803, 0.671, False),
        ("Random Forest", 0.955, 0.996, 0.525, 0.687, 0.803, 0.665, False),
        ("XGBoost", 0.955, 0.989, 0.526, 0.687, 0.802, 0.673, False),
        ("Logistic Regression + periodic", 0.955, 0.996, 0.524, 0.686, 0.828, 0.670, False),
        ("Random Forest + periodic", 0.955, 0.998, 0.525, 0.688, 0.844, 0.714, False),
        ("XGBoost + periodic (final)", 0.955, 0.995, 0.525, 0.687, 0.851, 0.718, True),
    ]
    features = [("is_bot_speed", 0.8708), ("device_accounts_total", 0.0641),
                ("time_since_signup", 0.0060), ("source_Direct", 0.0055),
                ("country_grouped_India", 0.0020)]
    return render_template("dashboard.html", active="dashboard", tiers=tiers, models=models,
                           features=features, total=total)


if __name__ == "__main__":
    app.run(debug=False)
