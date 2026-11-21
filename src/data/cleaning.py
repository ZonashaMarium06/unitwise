"""Clean every house: label each minute, then resample to hourly and daily kWh.

Run from the project root:  python -m src.data.cleaning
Total Usage_kW only; sub-meters are ignored (their layouts differ per house).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.loading import load_house, house_number, TOTAL_COL
from src.data.outage_audit import ZERO_KW

RAW_DIR = Path("data/raw/precon")
OUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

MIN_OUTAGE_MIN = 5       # zero for >= 5 min  -> outage
MAX_OUTAGE_MIN = 360     # zero for > 6 hours -> missing (masked)
MIN_FROZEN_MIN = 30      # identical non-zero value for >= 30 min -> frozen
MIN_VALID_SHARE = 0.8    # an hour needs >= 80% unmasked minutes
MIN_VALID_HOURS = 20     # a day needs >= 20 valid hours


def _run_length(mask: pd.Series) -> pd.Series:
    """For each True minute, the length of the unbroken True-run it belongs to."""
    run_id = (mask != mask.shift()).cumsum()
    return mask.astype(int).groupby(run_id).transform("sum")


def label_minutes(usage: pd.Series) -> pd.Series:
    zero = usage <= ZERO_KW
    zero_len = _run_length(zero)

    frozen = (usage.diff() == 0) & ~zero
    frozen_len = _run_length(frozen)

    label = pd.Series("valid", index=usage.index)
    label[frozen & (frozen_len >= MIN_FROZEN_MIN)] = "frozen"
    label[zero & (zero_len >= MIN_OUTAGE_MIN)] = "outage"
    label[zero & (zero_len > MAX_OUTAGE_MIN)] = "missing"
    return label


def to_hourly(usage: pd.Series, label: pd.Series) -> pd.DataFrame:
    kw = usage.where(label != "missing")   # masked minutes become NaN
    hourly = pd.DataFrame({
        "kwh": kw.resample("h").mean(),    # mean kW over 1 hour == kWh
        "valid_share": kw.notna().resample("h").mean(),
        "outage_share": (label == "outage").resample("h").mean(),
        "frozen_share": (label == "frozen").resample("h").mean(),
    })
    hourly.loc[hourly.valid_share < MIN_VALID_SHARE, "kwh"] = np.nan
    return hourly.astype("float32")


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    agg = hourly["kwh"].resample("D").agg(["mean", "count"])
    daily = pd.DataFrame({
        "kwh": agg["mean"] * 24,           # mean hourly kWh x 24 hours
        "valid_hours": agg["count"],
        "outage_hours": hourly["outage_share"].resample("D").sum(),
    })
    daily.loc[daily.valid_hours < MIN_VALID_HOURS, "kwh"] = np.nan
    return daily


def clean_house(path: Path):
    df = load_house(path)
    usage = df[TOTAL_COL]
    label = label_minutes(usage)
    hourly = to_hourly(usage, label)
    daily = to_daily(hourly)

    stats = {
        "house_id": house_number(path),
        "minutes": len(label),
        "pct_missing": round(100 * (label == "missing").mean(), 2),
        "pct_outage": round(100 * (label == "outage").mean(), 2),
        "pct_frozen": round(100 * (label == "frozen").mean(), 2),
        "hours": len(hourly),
        "hours_nan": int(hourly.kwh.isna().sum()),
        "days": len(daily),
        "days_nan": int(daily.kwh.isna().sum()),
        "daily_kwh_mean": round(float(daily.kwh.mean()), 2),
        "hourly_kwh_min": round(float(hourly.kwh.min()), 4),
    }
    return hourly, daily, stats


def main():
    (OUT_DIR / "hourly").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "daily").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    files = sorted(RAW_DIR.glob("House*.csv"), key=house_number)
    all_stats = []
    for i, path in enumerate(files, 1):
        hourly, daily, stats = clean_house(path)
        hid = stats["house_id"]
        hourly.to_parquet(OUT_DIR / "hourly" / f"house_{hid:02d}.parquet")
        daily.to_parquet(OUT_DIR / "daily" / f"house_{hid:02d}.parquet")
        all_stats.append(stats)
        print(f"[{i:2d}/{len(files)}] House {hid}: missing {stats['pct_missing']}%, "
              f"outage {stats['pct_outage']}%, days NaN {stats['days_nan']}")

    pd.DataFrame(all_stats).to_csv(REPORT_DIR / "cleaning_report.csv", index=False)
    print("Saved data/processed/ and reports/cleaning_report.csv")


if __name__ == "__main__":
    main()