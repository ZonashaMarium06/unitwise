"""Features for day-ahead hourly forecasting.

Forecast time = midnight. Every feature must be known by then, so every lag >= 24 h.
"""
import pandas as pd

HORIZON = 24

FEATURES = [
    "lag_24", "lag_48", "lag_72", "lag_168",
    "same_hour_7d_mean", "mean_24h", "mean_168h", "std_168h",
    "hour", "day_of_week", "is_weekend",
]


def make_features(hourly: pd.DataFrame, house_id: int) -> pd.DataFrame:
    s = hourly["kwh"]
    df = pd.DataFrame({"kwh": s})
    df["house_id"] = house_id

    # Lags: the same hour 1, 2, 3 and 7 days earlier
    for lag in (24, 48, 72, 168):
        df[f"lag_{lag}"] = s.shift(lag)

    # The same hour averaged over the previous 7 days
    df["same_hour_7d_mean"] = pd.concat(
        [s.shift(24 * k) for k in range(1, 8)], axis=1
    ).mean(axis=1)

    # Recent level and variability. shift FIRST, then roll: no leakage.
    past = s.shift(HORIZON)
    df["mean_24h"] = past.rolling(24, min_periods=12).mean()
    df["mean_168h"] = past.rolling(168, min_periods=84).mean()
    df["std_168h"] = past.rolling(168, min_periods=84).std()

    # Calendar (no month or day-of-year: see Concept 3)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.day_of_week
    df["is_weekend"] = (df.index.day_of_week >= 5).astype(int)
    return df