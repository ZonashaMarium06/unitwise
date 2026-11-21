"""Per-house anomaly detection with Isolation Forest.

Each hour is compared with what THIS house normally does at that hour
(its 7-day same-hour average). Isolation Forest then finds rare combinations.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

EPS = 0.02   # kWh: avoids dividing by ~0 in near-empty hours
ANOM_FEATURES = ["log_ratio", "z_dev", "kwh_rel", "hour_sin", "hour_cos", "is_weekend"]


def anomaly_features(feat: pd.DataFrame) -> pd.DataFrame:
    expected = feat["same_hour_7d_mean"]
    out = pd.DataFrame(index=feat.index)
    out["log_ratio"] = np.log((feat["kwh"] + EPS) / (expected + EPS))
    out["z_dev"] = ((feat["kwh"] - expected) / (feat["std_168h"] + EPS)).clip(-10, 10)
    out["kwh_rel"] = (feat["kwh"] / (feat["mean_168h"] + EPS)).clip(0, 10)
    out["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
    out["is_weekend"] = feat["is_weekend"]
    return out


def detect(hourly, feat, train_end, contamination=0.01, seed=42):
    """Fit on hours before train_end, score every valid hour."""
    X = anomaly_features(feat)
    ok = X.notna().all(axis=1) & (hourly["outage_share"] < 0.5)

    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=seed)
    model.fit(X.loc[ok & (X.index < train_end), ANOM_FEATURES])

    res = pd.DataFrame({"kwh": feat["kwh"], "expected": feat["same_hour_7d_mean"]})
    res["flag"] = False
    res["score"] = np.nan
    res.loc[ok, "score"] = -model.decision_function(X.loc[ok, ANOM_FEATURES])  # higher = stranger
    res.loc[ok, "flag"] = model.predict(X.loc[ok, ANOM_FEATURES]) == -1
    res["outage"] = hourly["outage_share"] >= 0.5
    return res


def explain(res):
    """Attach a plain-language reason to every flagged hour."""
    f = res[res.flag].copy()
    ratio = (f.kwh + EPS) / (f.expected + EPS)
    direction = np.where(f.kwh > f.expected, "higher", "lower")
    f["reason"] = [
        f"{h:02d}:00 usage {k:.2f} kWh vs usual {e:.2f} kWh ({r:.1f}x, {d})"
        for h, k, e, r, d in zip(f.index.hour, f.kwh, f.expected, ratio, direction)
    ]
    return f


def to_events(flagged):
    """Merge consecutive flagged hours into single alerts."""
    new_event = flagged.index.to_series().diff() != pd.Timedelta("1h")
    return (flagged.assign(event=new_event.cumsum().to_numpy())
            .groupby("event")
            .agg(start=("kwh", lambda s: s.index.min()), hours=("kwh", "size"),
                 peak_kwh=("kwh", "max"), expected_kwh=("expected", "mean")))