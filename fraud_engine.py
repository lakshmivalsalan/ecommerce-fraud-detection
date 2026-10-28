"""
Scoring engine for the e-commerce fraud detection web app.

It rebuilds exactly the features the final XGBoost model was trained on
(see fraud_detection_rebuild.ipynb) and turns the fraud probability into
High / Medium / Low risk tiers.

Hybrid design: the ML model gives the fraud probability and risk tier.
A business rule is then applied on top: a purchase made within 1 minute
of signup on a device shared by 10 or more accounts is always High risk.
The model's probability is never changed; the rule only raises the tier.

Needs these files in the same folder:
    fraud_model_xgb_final.json   final XGBoost model (saved with save_model)
    model_config_final.json      feature order, top countries, tier cut-offs
    IpAddress_to_Country.csv     IP range -> country lookup
    Fraud_Data.csv               (optional) used for the demo batch
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

BASE = Path(__file__).parent

REQUIRED = [
    "signup_time", "purchase_time", "purchase_value", "device_id",
    "source", "browser", "sex", "age", "ip_address",
]
TIERS = ["High", "Medium", "Low"]
TIER_ACTION = {
    "High": "Block or flag automatically",
    "Medium": "Send to manual review",
    "Low": "Let through",
}

# Business rule (applied on top of the model)
RULE_MAX_SECONDS = 60      # purchase within 1 minute of signup
RULE_MIN_ACCOUNTS = 10     # device shared by 10 or more accounts
RULE_TEXT = "Flagged by rule: rapid purchase on a heavily shared device"

# Results of the final model on the locked test set (from the notebook).
TEST_TIERS = {"High": (1160, 1155), "Medium": (1700, 364), "Low": (19807, 603)}


def rule_hit(gap, accounts):
    """True where the business rule applies."""
    gap = np.asarray(gap, dtype=float)
    accounts = np.asarray(accounts, dtype=float)
    return (gap <= RULE_MAX_SECONDS) & (accounts >= RULE_MIN_ACCOUNTS)


def time_text(gap):
    """Readable time between signup and purchase, e.g. '45 seconds', '1 minute', '3.0 days'."""
    if gap < 60:
        n = int(round(gap))
        return f"{n} second" + ("" if n == 1 else "s")
    if gap < 3600:
        n = int(round(gap / 60))
        return f"{n} minute" + ("" if n == 1 else "s")
    if gap < 86400:
        n = int(round(gap / 3600))
        return f"{n} hour" + ("" if n == 1 else "s")
    return f"{gap / 86400:,.1f} days"


class FraudEngine:
    def __init__(self):
        self.model = XGBClassifier()
        self.model.load_model(str(BASE / "fraud_model_xgb_final.json"))
        self.config = json.loads((BASE / "model_config_final.json").read_text())
        ip = pd.read_csv(BASE / "IpAddress_to_Country.csv").sort_values("lower_bound_ip_address")
        self.lower = ip["lower_bound_ip_address"].values
        self.upper = ip["upper_bound_ip_address"].values
        self.names = ip["country"].values

    # ------------------------------------------------------------ features
    def find_country(self, ip_numbers):
        ip_numbers = np.asarray(ip_numbers, dtype=float)
        pos = np.searchsorted(self.lower, ip_numbers, side="right") - 1
        safe = np.clip(pos, 0, None)
        found = (pos >= 0) & (ip_numbers <= self.upper[safe])
        return np.where(found, self.names[safe], "Unknown")

    def make_features(self, raw):
        """Rebuild exactly the features the model was trained on."""
        d = raw.copy()
        d["signup_time"] = pd.to_datetime(d["signup_time"], errors="coerce")
        d["purchase_time"] = pd.to_datetime(d["purchase_time"], errors="coerce")
        d["ip_address"] = pd.to_numeric(d["ip_address"], errors="coerce").fillna(-1)

        d["time_since_signup"] = (d["purchase_time"] - d["signup_time"]).dt.total_seconds()
        d["is_bot_speed"] = (d["time_since_signup"] <= 1).astype(int)
        d["purchase_hour"] = d["purchase_time"].dt.hour
        d["purchase_dayofweek"] = d["purchase_time"].dt.dayofweek
        d["signup_hour"] = d["signup_time"].dt.hour

        d["country"] = self.find_country(d["ip_address"].values)
        keep = d["country"].isin(self.config["top_countries"]) | (d["country"] == "Unknown")
        d["country_grouped"] = d["country"].where(keep, "Other")

        cols = ["purchase_value", "age", "time_since_signup", "is_bot_speed", "purchase_hour",
                "purchase_dayofweek", "signup_hour", "device_accounts_total",
                "source", "browser", "sex", "country_grouped"]
        enc = pd.get_dummies(d[cols], columns=["source", "browser", "sex", "country_grouped"], dtype=int)
        enc = enc.reindex(columns=self.config["feature_columns"], fill_value=0)
        return d, enc

    def tier_of(self, prob):
        prob = np.asarray(prob)
        return np.where(prob >= self.config["high_threshold"], "High",
                        np.where(prob >= self.config["medium_threshold"], "Medium", "Low"))

    @staticmethod
    def reason_text(gap, accounts, by_rule=False):
        parts = []
        if by_rule:
            parts.append(RULE_TEXT)
        if gap <= 1:
            parts.append("Bought within 1 second of signup")
        elif gap < 3600:
            parts.append(f"Bought {time_text(gap)} after signup")
        if accounts >= 2:
            parts.append(f"{int(accounts)} accounts on one device")
        return "; ".join(parts) if parts else "No strong signal"

    # ------------------------------------------------------------ scoring
    def score_batch(self, raw):
        raw = raw.reset_index(drop=True)
        missing = [c for c in REQUIRED if c not in raw.columns]
        if missing:
            raise ValueError("The file is missing these columns: " + ", ".join(missing))
        if "user_id" not in raw.columns:
            raw.insert(0, "user_id", np.arange(1, len(raw) + 1))

        # Periodic review: how many accounts share each device in this batch.
        raw["device_accounts_total"] = raw.groupby("device_id")["device_id"].transform("count")

        d, enc = self.make_features(raw)
        prob = self.model.predict_proba(enc.astype(float))[:, 1]
        gaps = d["time_since_signup"].values
        accounts = d["device_accounts_total"].values

        model_tier = self.tier_of(prob)
        by_rule = rule_hit(gaps, accounts) & (model_tier != "High")
        final_tier = np.where(by_rule, "High", model_tier)

        out = pd.DataFrame({
            "user_id": d["user_id"],
            "risk_score": prob.round(4),
            "risk_tier": final_tier,
            "decided_by": np.where(by_rule, "Rule", "Model"),
            "reason": [self.reason_text(g, n, r) for g, n, r in zip(gaps, accounts, by_rule)],
            "purchase_value": d["purchase_value"],
            "seconds_since_signup": d["time_since_signup"],
            "accounts_on_device": d["device_accounts_total"],
            "country": d["country"],
            "source": d["source"],
            "browser": d["browser"],
            "age": d["age"],
        })
        if "class" in raw.columns:
            out["actual_fraud"] = raw["class"].values
        return out

    def score_one(self, signup, purchase, value, age, sex, browser, source, ip_num, devices):
        row = pd.DataFrame({
            "signup_time": [signup], "purchase_time": [purchase],
            "purchase_value": [value], "age": [age], "sex": [sex],
            "browser": [browser], "source": [source], "ip_address": [ip_num],
            "device_accounts_total": [int(devices)],
        })
        d, enc = self.make_features(row)
        prob = float(self.model.predict_proba(enc.astype(float))[:, 1][0])
        gap = float(d["time_since_signup"].iloc[0])

        tier = str(self.tier_of([prob])[0])
        by_rule = bool(rule_hit([gap], [devices])[0]) and tier != "High"
        if by_rule:
            tier = "High"

        reasons = []
        if by_rule:
            reasons.append(("warn", RULE_TEXT))
        if gap <= 1:
            reasons.append(("warn", "Bought within 1 second of signup"))
        elif gap < 3600:
            reasons.append(("warn", f"Bought {time_text(gap)} after signup"))
        else:
            reasons.append(("ok", f"Bought {time_text(gap)} after signup"))
        if devices >= 2:
            reasons.append(("warn", f"Device used by {int(devices)} accounts"))
        else:
            reasons.append(("ok", "Device used by this account only"))
        reasons.append(("info", f"IP address is from {d['country'].iloc[0]}"))

        return {"prob": prob, "tier": tier, "action": TIER_ACTION[tier], "reasons": reasons,
                "by_rule": by_rule}

    @staticmethod
    def load_demo():
        path = BASE / "Fraud_Data.csv"
        if not path.exists():
            return None
        full = pd.read_csv(path)
        devices = full["device_id"].drop_duplicates().sample(1500, random_state=7)
        return full[full["device_id"].isin(devices)].reset_index(drop=True)


def ip_to_number(text):
    """Accept '66.249.66.1' or a plain number, as stored in the dataset."""
    text = (text or "").strip()
    parts = text.split(".")
    if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        a, b, c, d = (int(p) for p in parts)
        return a * 256**3 + b * 256**2 + c * 256 + d
    try:
        return float(text)
    except ValueError:
        return None