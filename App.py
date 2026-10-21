"""
Account risk review: Streamlit app for the e-commerce fraud mini project.

Needs these files in the same folder as this script:
    fraud_model_xgb_final.json  (final XGBoost model, saved with save_model)
    model_config_final.json     (feature order, kept countries, thresholds)
    IpAddress_to_Country.csv    (IP range to country lookup)
Optional, for the demo batch:
    Fraud_Data.csv

Run with:  python -m streamlit run App.py
"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

BASE = Path(__file__).parent

TIERS = ["High", "Medium", "Low"]
TIER_CLASS = {"High": "high", "Medium": "med", "Low": "low"}
TIER_COLOR = {"High": "#E5484D", "Medium": "#F2A20C", "Low": "#2FB383"}
TIER_INK = {"High": "#B42318", "Medium": "#8A5300", "Low": "#087443"}
TIER_ACTION = {
    "High": "Block or flag automatically",
    "Medium": "Send to manual review",
    "Low": "Let through",
}

# Held-out test set results, shown before any batch is loaded.
# tier: (accounts, real frauds)
REFERENCE = {"High": (1160, 1155), "Medium": (1700, 364), "Low": (19807, 603)}

REQUIRED = [
    "signup_time", "purchase_time", "purchase_value", "device_id",
    "source", "browser", "sex", "age", "ip_address",
]

st.set_page_config(page_title="Account risk review", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:wght@400;600&display=swap');
:root {
  --ink:#0E1B3D; --muted:#56627C; --line:#E1E6F0; --brand:#3B5BDB; --wash:#EAEFF8;
  --high:#E5484D; --high-ink:#B42318; --high-tint:#FDECEC;
  --med:#F2A20C;  --med-ink:#8A5300;  --med-tint:#FEF3DC;
  --low:#2FB383;  --low-ink:#087443;  --low-tint:#E3F6EE;
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th {
  font-family: 'Source Sans 3', 'Segoe UI', system-ui, sans-serif;
}
.rr-title, .rr-h, .rr-num, .rr-gauge-num {
  font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif !important;
}
.stAppDeployButton, [data-testid="stToolbar"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 2.6rem; padding-bottom: 4rem; }

.rr-brand { display: flex; align-items: center; gap: 0.9rem; }
.rr-mark { width: 44px; height: 44px; border-radius: 12px; background: var(--brand);
  display: flex; align-items: center; justify-content: center; flex: none; }
.rr-brand h1.rr-title { font-size: 2.5rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--ink); margin: 0; padding: 0; line-height: 1.05; }
.stApp p.rr-lede { color: var(--muted); font-size: 1.1rem; max-width: 40rem; margin: 0.9rem 0 1.8rem 0; }

button[data-baseweb="tab"] { font-weight: 600; font-size: 1.02rem; }
[data-baseweb="tab-list"] { gap: 1.6rem; }

[data-testid="stFileUploaderDropzone"] {
  background: #FFFFFF; border: 1.5px dashed #AEB9D6; border-radius: 14px; padding: 1.1rem 1.2rem;
}

.rr-h { font-size: 1.3rem; font-weight: 700; color: var(--ink); letter-spacing: -0.01em; margin: 0 0 0.4rem 0; }
.rr-meta { color: var(--muted); font-size: 0.98rem; margin: 0.15rem 0 0 0; }

.rr-ledger { margin: 1.6rem 0 0.6rem 0; }
.rr-bar { display: flex; gap: 4px; height: 54px; margin: 1rem 0 1.3rem 0; }
.rr-seg { min-width: 8px; height: 100%; }
.rr-seg:first-child { border-radius: 12px 4px 4px 12px; }
.rr-seg:last-child { border-radius: 4px 12px 12px 4px; }
.rr-seg:only-child { border-radius: 12px; }
.rr-tiers { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.8rem; }
.rr-t { border-left: 4px solid var(--c); padding-left: 1rem; }
.rr-pill { display: inline-block; font-weight: 600; font-size: 0.88rem; padding: 0.12rem 0.7rem;
  border-radius: 999px; }
.rr-pill.high { background: var(--high-tint); color: var(--high-ink); }
.rr-pill.med  { background: var(--med-tint);  color: var(--med-ink); }
.rr-pill.low  { background: var(--low-tint);  color: var(--low-ink); }
.rr-num { font-size: 2.7rem; font-weight: 700; line-height: 1.15; color: var(--ink); margin-top: 0.3rem; }
.rr-sub { color: var(--muted); font-size: 0.97rem; line-height: 1.4; }
.rr-act { color: var(--ink); font-weight: 600; font-size: 0.97rem; margin-top: 0.15rem; }

.rr-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.4rem 0 0.8rem 0; }
.rr-chip { background: var(--wash); color: var(--ink); border-radius: 999px;
  padding: 0.25rem 0.85rem; font-size: 0.95rem; }
.stApp p.rr-note { color: var(--muted); font-size: 0.95rem; }

.rr-scale { display: flex; height: 18px; border-radius: 9px; overflow: hidden; margin: 0.8rem 0 0.3rem 0; }
.rr-scale-labels { display: flex; color: var(--muted); font-size: 0.9rem; }

@media (max-width: 720px) {
  .rr-tiers { grid-template-columns: 1fr; gap: 1.2rem; }
  .rr-brand h1.rr-title { font-size: 1.9rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- engine
@st.cache_resource
def load_assets():
    # Prefer XGBoost's own file format: it loads across XGBoost versions.
    native = BASE / "fraud_model_xgb_final.json"
    if native.exists():
        model = XGBClassifier()
        model.load_model(str(native))
    else:
        model = joblib.load(BASE / "fraud_model_xgb_final.pkl")
    config = json.loads((BASE / "model_config_final.json").read_text())
    ip = pd.read_csv(BASE / "IpAddress_to_Country.csv").sort_values("lower_bound_ip_address")
    lookup = (
        ip["lower_bound_ip_address"].values,
        ip["upper_bound_ip_address"].values,
        ip["country"].values,
    )
    return model, config, lookup


def find_country(ip_numbers, lookup):
    lower, upper, names = lookup
    ip_numbers = np.asarray(ip_numbers, dtype=float)
    pos = np.searchsorted(lower, ip_numbers, side="right") - 1
    safe = np.clip(pos, 0, None)
    found = (pos >= 0) & (ip_numbers <= upper[safe])
    return np.where(found, names[safe], "Unknown")


def make_features(raw, config, lookup):
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

    d["country"] = find_country(d["ip_address"].values, lookup)
    keep = d["country"].isin(config["top_countries"]) | (d["country"] == "Unknown")
    d["country_grouped"] = d["country"].where(keep, "Other")

    keep_cols = [
        "purchase_value", "age", "time_since_signup", "is_bot_speed", "purchase_hour",
        "purchase_dayofweek", "signup_hour", "device_accounts_total",
        "source", "browser", "sex", "country_grouped",
    ]
    enc = pd.get_dummies(
        d[keep_cols], columns=["source", "browser", "sex", "country_grouped"], dtype=int
    )
    enc = enc.reindex(columns=config["feature_columns"], fill_value=0)
    return d, enc


def tier_of(prob, config):
    return np.where(
        prob >= config["high_threshold"], "High",
        np.where(prob >= config["medium_threshold"], "Medium", "Low"),
    )


def why_text(gap, accounts):
    parts = []
    if gap <= 1:
        parts.append("1-second purchase")
    if accounts >= 2:
        parts.append(f"{int(accounts)} accounts on device")
    return ", ".join(parts) if parts else "No strong signal"


@st.cache_data(show_spinner="Scoring accounts")
def score_batch(raw: pd.DataFrame):
    model, config, lookup = load_assets()
    raw = raw.reset_index(drop=True)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if "user_id" not in raw.columns:
        raw.insert(0, "user_id", np.arange(1, len(raw) + 1))

    # Periodic review: how many accounts share each device in this batch.
    raw["device_accounts_total"] = raw.groupby("device_id")["device_id"].transform("count")

    d, enc = make_features(raw, config, lookup)
    prob = model.predict_proba(enc.astype(float))[:, 1]
    out = pd.DataFrame(
        {
            "user_id": d["user_id"],
            "risk_score": prob,
            "risk_tier": tier_of(prob, config),
            "why": [
                why_text(g, n)
                for g, n in zip(d["time_since_signup"].values, d["device_accounts_total"].values)
            ],
            "purchase_value": d["purchase_value"],
            "seconds_since_signup": d["time_since_signup"],
            "accounts_on_device": d["device_accounts_total"],
            "country": d["country"],
            "source": d["source"],
            "browser": d["browser"],
            "age": d["age"],
        }
    )
    if "class" in raw.columns:
        out["actual_fraud"] = raw["class"].values
    return out


def load_demo():
    path = BASE / "Fraud_Data.csv"
    if not path.exists():
        return None
    full = pd.read_csv(path)
    devices = full["device_id"].drop_duplicates().sample(1500, random_state=7)
    return full[full["device_id"].isin(devices)].reset_index(drop=True)


def ip_to_number(text):
    parts = text.strip().split(".")
    if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        a, b, c, d = (int(p) for p in parts)
        return a * 256**3 + b * 256**2 + c * 256 + d
    if text.strip().isdigit():
        return int(text.strip())
    return None


# ---------------------------------------------------------------- visual pieces
def ledger_html(heading, note, counts, fraud_in_tier=None):
    """Proportional bar plus one column per tier."""
    total = max(sum(counts.values()), 1)
    segs = "".join(
        f'<div class="rr-seg" style="width:{max(counts[t] / total * 100, 1.0):.2f}%;'
        f'background:{TIER_COLOR[t]}"></div>'
        for t in TIERS
    )
    cols = []
    for t in TIERS:
        extra = ""
        if fraud_in_tier is not None and counts[t] > 0:
            extra = f" {fraud_in_tier[t] / counts[t] * 100:.1f}% of these were real fraud."
        cols.append(
            f'<div class="rr-t" style="--c:{TIER_COLOR[t]}">'
            f'<span class="rr-pill {TIER_CLASS[t]}">{t} risk</span>'
            f'<div class="rr-num">{counts[t]:,}</div>'
            f'<div class="rr-sub">{counts[t] / total * 100:.1f}% of accounts.{extra}</div>'
            f'<div class="rr-act">{TIER_ACTION[t]}</div></div>'
        )
    return (
        f'<div class="rr-ledger"><div class="rr-h">{heading}</div>'
        f'<div class="rr-meta">{note}</div>'
        f'<div class="rr-bar" role="img" aria-label="Share of accounts by risk tier">{segs}</div>'
        f'<div class="rr-tiers">{"".join(cols)}</div></div>'
    )


def gauge_svg(prob, cfg):
    """Half-circle gauge: zones for Low, Medium, High and a needle at the score."""
    cx, cy, r = 150, 150, 118
    lo, hi = cfg["medium_threshold"], cfg["high_threshold"]

    def pt(frac, radius):
        ang = math.pi * (1 - frac)
        return cx + radius * math.cos(ang), cy - radius * math.sin(ang)

    def arc(f0, f1, color):
        x0, y0 = pt(f0, r)
        x1, y1 = pt(f1, r)
        return (
            f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}" '
            f'stroke="{color}" stroke-width="22" fill="none"/>'
        )

    active = prob is not None
    colors = (
        (TIER_COLOR["Low"], TIER_COLOR["Medium"], TIER_COLOR["High"]) if active
        else ("#DCE3F0", "#DCE3F0", "#DCE3F0")
    )
    parts = [arc(0, lo, colors[0]), arc(lo, hi, colors[1]), arc(hi, 1, colors[2])]

    for frac, label in [(0, "0"), (lo, f"{lo:g}"), (hi, f"{hi:g}"), (1, "1")]:
        x, y = pt(frac, r + 24)
        parts.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="12" '
            f'fill="#56627C">{label}</text>'
        )

    if active:
        tier = str(tier_of(np.array([prob]), cfg)[0])
        nx, ny = pt(min(max(prob, 0), 1), r - 34)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#0E1B3D" '
            f'stroke-width="5" stroke-linecap="round"/><circle cx="{cx}" cy="{cy}" r="10" fill="#0E1B3D"/>'
        )
        score, label, label_color = f"{prob:.2f}", f"{tier} risk", TIER_INK[tier]
    else:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="10" fill="#AEB9D6"/>')
        score, label, label_color = "&#8211;", "Score an account to see its risk", "#56627C"

    return (
        '<svg viewBox="0 0 300 235" width="100%" role="img" '
        f'aria-label="Fraud score gauge: {label}" xmlns="http://www.w3.org/2000/svg">'
        + "".join(parts)
        + f'<text x="{cx}" y="198" text-anchor="middle" font-size="40" font-weight="700" '
        f'fill="#0E1B3D" class="rr-gauge-num">{score}</text>'
        f'<text x="{cx}" y="224" text-anchor="middle" font-size="17" font-weight="600" '
        f'fill="{label_color}">{label}</text></svg>'
    )


# ---------------------------------------------------------------- header
st.markdown(
    '<div class="rr-brand"><div class="rr-mark">'
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6l7-3z"/>'
    '<path d="M9 12l2 2 4-4"/></svg></div><h1 class="rr-title">Account risk review</h1></div>'
    '<p class="rr-lede">Sort the accounts from a review period into High, Medium and Low risk, '
    "so you know which ones to act on first.</p>",
    unsafe_allow_html=True,
)

try:
    load_assets()
except Exception as err:  # missing files, version mismatch...
    st.error(
        "The model files could not be loaded. Put fraud_model_xgb_final.json, "
        "model_config_final.json and IpAddress_to_Country.csv in the same folder as App.py."
    )
    st.caption(f"Details: {err}")
    st.stop()

tab_batch, tab_single, tab_about = st.tabs(["Batch review", "Single transaction", "About the model"])

# ---------------------------------------------------------------- batch tab
with tab_batch:
    left, right = st.columns([3, 1], gap="medium", vertical_alignment="center")
    with left:
        upload = st.file_uploader(
            "Transactions (CSV)", type="csv", label_visibility="collapsed",
            help="Needed columns: " + ", ".join(REQUIRED) + ". user_id is optional.",
        )
    with right:
        if st.button("Try a demo batch", type="primary", use_container_width=True):
            st.session_state["demo"] = True
        st.caption("About 1,700 accounts from Fraud_Data.csv")

    raw = None
    if upload is not None:
        st.session_state["demo"] = False
        raw = pd.read_csv(upload)
    elif st.session_state.get("demo"):
        raw = load_demo()
        if raw is None:
            st.warning("Fraud_Data.csv was not found next to App.py, so the demo cannot load.")

    if raw is None:
        ref_counts = {t: REFERENCE[t][0] for t in TIERS}
        ref_fraud = {t: REFERENCE[t][1] for t in TIERS}
        st.markdown(
            ledger_html(
                "What the model found on accounts it had never seen",
                "Held-out test set of 22,667 accounts. Upload your own CSV above to see the same view for your batch.",
                ref_counts, ref_fraud,
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "CSV columns needed: " + ", ".join(REQUIRED) + ". A user_id column is optional."
        )
    else:
        try:
            scored = score_batch(raw)
        except ValueError as err:
            st.error(str(err) + ". Add the missing columns and upload again.")
            st.stop()

        counts = {t: int((scored["risk_tier"] == t).sum()) for t in TIERS}
        fraud_in_tier = None
        if "actual_fraud" in scored.columns:
            fraud_in_tier = {
                t: int(((scored["risk_tier"] == t) & (scored["actual_fraud"] == 1)).sum()) for t in TIERS
            }

        note = f"{len(scored):,} accounts scored."
        if st.session_state.get("demo") and upload is None:
            note += " Demo data comes from the training dataset, so use it to see how the app works, not to judge accuracy."
        st.markdown(
            ledger_html("Your batch at a glance", note, counts, fraud_in_tier),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="rr-h" style="margin:1.6rem 0 0.5rem 0">Review queue</div>', unsafe_allow_html=True)
        labels = {t: f"{t} risk ({counts[t]:,})" for t in TIERS}
        if hasattr(st, "segmented_control"):
            pick = st.segmented_control(
                "Show accounts in", TIERS, default="High", format_func=lambda t: labels[t],
                label_visibility="collapsed", key="tier_pick",
            ) or "High"
        else:
            pick = st.radio(
                "Show accounts in", TIERS, format_func=lambda t: labels[t],
                horizontal=True, label_visibility="collapsed",
            )

        view = scored[scored["risk_tier"] == pick].sort_values("risk_score", ascending=False)
        drop = ["risk_tier"] + (["actual_fraud"] if "actual_fraud" in view else [])
        shown = view.drop(columns=drop)

        if shown.empty:
            st.write(f"No accounts fall in the {pick} tier for this batch.")
        else:
            st.dataframe(
                shown,
                hide_index=True,
                height=min(460, 38 * (len(shown) + 1)),
                column_order=[
                    "user_id", "risk_score", "why", "purchase_value",
                    "seconds_since_signup", "accounts_on_device", "country",
                ],
                column_config={
                    "user_id": st.column_config.NumberColumn("User", format="%d", width="small"),
                    "risk_score": st.column_config.ProgressColumn(
                        "Fraud score", min_value=0.0, max_value=1.0, format="%.2f", width="small"
                    ),
                    "why": st.column_config.TextColumn("Why flagged", width=300),
                    "purchase_value": st.column_config.NumberColumn("Value", width="small"),
                    "seconds_since_signup": st.column_config.NumberColumn(
                        "Time to buy (s)", format="%d", width="small"
                    ),
                    "accounts_on_device": st.column_config.NumberColumn("On device", width="small"),
                    "country": "Country",
                    "source": "Source",
                    "browser": "Browser",
                    "age": "Age",
                },
            )

        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            st.download_button(
                f"Download {pick} risk list",
                view.to_csv(index=False).encode(),
                file_name=f"{pick.lower()}_risk_accounts.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download all accounts",
                scored.to_csv(index=False).encode(),
                file_name="all_scored_accounts.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---------------------------------------------------------------- single tab
with tab_single:
    _, cfg_s, _ = load_assets()
    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col:
        st.markdown('<div class="rr-h">Score one account</div>'
                    '<div class="rr-meta">Enter what the review period shows for it.</div>',
                    unsafe_allow_html=True)
        with st.form("single", border=False):
            a1, a2 = st.columns(2)
            value = a1.number_input("Purchase value", min_value=0, value=40, step=1)
            age = a2.number_input("Age", min_value=18, max_value=100, value=33, step=1)
            b1, b2 = st.columns(2)
            sex = b1.selectbox("Sex", ["M", "F"])
            browser = b2.selectbox("Browser", ["Chrome", "IE", "Safari", "FireFox", "Opera"])
            c1, c2 = st.columns(2)
            source = c1.selectbox("Source", ["SEO", "Ads", "Direct"])
            ip_text = c2.text_input("IP address", value="66.249.66.1")
            d1_, d2_ = st.columns(2)
            signup_text = d1_.text_input(
                "Signup time", value="2015-05-01 09:30:00", help="Format: YYYY-MM-DD HH:MM:SS"
            )
            purchase_text = d2_.text_input(
                "Purchase time", value="2015-06-12 18:45:00", help="Format: YYYY-MM-DD HH:MM:SS"
            )
            devices = st.number_input(
                "Accounts that used this device in the period",
                min_value=1, value=1, step=1,
                help="Count every account on the same device, including this one.",
            )
            go = st.form_submit_button("Score this account", type="primary")

    with result_col:
        result_slot = st.container()
        prob = None
        error = None
        reasons = []
        if go:
            ip_num = ip_to_number(ip_text)
            signup = pd.to_datetime(signup_text, errors="coerce")
            purchase = pd.to_datetime(purchase_text, errors="coerce")
            if ip_num is None:
                error = "The IP address should look like 66.249.66.1."
            elif pd.isna(signup) or pd.isna(purchase):
                error = "Times should look like 2015-05-01 09:30:00 (year-month-day hour:minute:second)."
            elif purchase < signup:
                error = "The purchase time is before the signup time. Check both dates."
            else:
                model, config, lookup = load_assets()
                row = pd.DataFrame(
                    {
                        "signup_time": [signup], "purchase_time": [purchase],
                        "purchase_value": [value], "age": [age], "sex": [sex],
                        "browser": [browser], "source": [source], "ip_address": [ip_num],
                        "device_accounts_total": [int(devices)],
                    }
                )
                d, enc = make_features(row, config, lookup)
                prob = float(model.predict_proba(enc.astype(float))[:, 1][0])
                gap = float(d["time_since_signup"].iloc[0])
                if gap <= 1:
                    reasons.append("Bought within 1 second of signup")
                else:
                    reasons.append(f"Bought {gap / 86400:,.1f} days after signup")
                reasons.append(
                    f"Device used by {int(devices)} accounts" if devices >= 2 else "Device used by this account only"
                )
                reasons.append(f"IP address maps to {d['country'].iloc[0]}")

        with result_slot:
            if error:
                st.error(error)
            st.markdown(gauge_svg(prob, cfg_s), unsafe_allow_html=True)
            if prob is not None:
                tier = str(tier_of(np.array([prob]), cfg_s)[0])
                st.markdown(
                    f'<div class="rr-act" style="text-align:center;margin-bottom:0.9rem">{TIER_ACTION[tier]}</div>'
                    '<div class="rr-h" style="font-size:1.05rem">What the model saw</div>'
                    '<div class="rr-chips">' + "".join(f'<span class="rr-chip">{r}</span>' for r in reasons) + "</div>",
                    unsafe_allow_html=True,
                )
                if tier == "Low":
                    st.markdown(
                        '<p class="rr-note">Low risk is not proof of a genuine customer. '
                        "Some fraud shows none of the signals this model can read.</p>",
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------- about tab
with tab_about:
    _, cfg, _ = load_assets()
    lo, hi = cfg["medium_threshold"], cfg["high_threshold"]

    st.markdown('<div class="rr-h">How it decides</div>', unsafe_allow_html=True)
    st.write(
        "The model, an XGBoost classifier, reads two strong signals in this data: purchases "
        "made within one second of signup, and devices shared by several accounts. "
        "It turns them into a fraud score between 0 and 1, and the score decides the tier."
    )
    st.markdown(
        '<div class="rr-scale">'
        f'<div style="width:{lo * 100:.1f}%;background:{TIER_COLOR["Low"]}"></div>'
        f'<div style="width:{(hi - lo) * 100:.1f}%;background:{TIER_COLOR["Medium"]}"></div>'
        f'<div style="width:{(1 - hi) * 100:.1f}%;background:{TIER_COLOR["High"]}"></div></div>'
        '<div class="rr-scale-labels">'
        f'<div style="width:{lo * 100:.1f}%">0</div>'
        f'<div style="width:{(hi - lo) * 100:.1f}%">{lo:g}</div>'
        f'<div style="width:{(1 - hi) * 100:.1f}%">{hi:g}</div></div>'
        '<p class="rr-note" style="margin-top:0.6rem">Below '
        f'{lo:g} is Low risk, from {lo:g} to just under {hi:g} is Medium risk, and {hi:g} or more is High risk.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rr-h" style="margin-top:1.8rem">Results on the held-out test set</div>',
                unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(
            {
                "Tier": TIERS,
                "Accounts": [f"{REFERENCE[t][0]:,}" for t in TIERS],
                "Real frauds": [f"{REFERENCE[t][1]:,}" for t in TIERS],
                "Fraud rate": ["99.6%", "21.4%", "3.0%"],
            }
        ),
        hide_index=True,
    )
    st.write(
        "High and Medium together catch 71.6% of all fraud (precision 0.53). "
        "PR-AUC on the test set is 0.721."
    )

    st.markdown('<div class="rr-h" style="margin-top:1.8rem">What it cannot do</div>', unsafe_allow_html=True)
    st.markdown(
        "- About 28% of the fraud looks identical to normal activity on every feature "
        "available, so it lands in Low risk.\n"
        "- Device counts come from the whole batch you upload. This is a review tool for a "
        "period of transactions, not a real-time check at the moment of purchase.\n"
        "- The dataset looks simulated, so these results show the method works, "
        "not how it would perform on a real shop's data."
    )