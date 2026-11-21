"""Data-quality audit for all PRECON houses.

Run from the project root:  python -m src.data.audit
"""
from pathlib import Path
import pandas as pd
from src.data.loading import load_house, house_number, TOTAL_COL

RAW_DIR = Path("data/raw/precon")
REPORT_DIR = Path("reports")
ONE_MIN = pd.Timedelta("1min")


def audit_house(path: Path):
    df = load_house(path)
    idx = df.index
    num_cols = [c for c in df.columns if c != "house_id"]
    sub_cols = [c for c in num_cols if c != TOTAL_COL]
    usage = df[TOTAL_COL]

    start, end = idx.min(), idx.max()
    expected = int((end - start) / ONE_MIN) + 1
    unique_ts = pd.Series(idx.unique()).sort_values().reset_index(drop=True)
    missing_minutes = expected - len(unique_ts)

    # Gaps: places where consecutive timestamps are more than 1 minute apart
    delta = unique_ts.diff()
    gap_mask = delta > ONE_MIN
    gaps = pd.DataFrame({
        "house_id": house_number(path),
        "last_seen_before": unique_ts.shift(1)[gap_mask].values,
        "next_seen_after": unique_ts[gap_mask].values,
        "missing_minutes": (delta[gap_mask] / ONE_MIN - 1).astype(int).values,
    })

    summary = {
        "house_id": house_number(path),
        "start": start,
        "end": end,
        "days_span": round((end - start).total_seconds() / 86400, 1),
        "rows": len(df),
        "expected_rows": expected,
        "coverage_pct": round(100 * len(unique_ts) / expected, 2),
        "missing_minutes": missing_minutes,
        "n_gaps": len(gaps),
        "longest_gap_hours": round(gaps.missing_minutes.max() / 60, 1) if len(gaps) else 0,
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "nan_cells": int(df[num_cols].isna().sum().sum()),
        "negative_values": int((df[num_cols] < 0).sum().sum()),
        "zero_usage_minutes": int((usage == 0).sum()),
        "repeated_value_pct": round(100 * (usage.diff() == 0).mean(), 2),
        "mean_kW": round(float(usage.mean()), 3),
        "max_kW": round(float(usage.max()), 3),
        "submeter_share_pct": round(100 * float(df[sub_cols].sum().sum() / usage.sum()), 1),
        "columns": ",".join(num_cols),
    }
    return summary, gaps


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    files = sorted(RAW_DIR.glob("House*.csv"), key=house_number)
    print(f"Auditing {len(files)} files...")

    summaries, all_gaps = [], []
    for i, path in enumerate(files, 1):
        summary, gaps = audit_house(path)
        summaries.append(summary)
        all_gaps.append(gaps)
        print(f"[{i:2d}/{len(files)}] {path.name}: coverage {summary['coverage_pct']}%")

    pd.DataFrame(summaries).to_csv(REPORT_DIR / "audit_summary.csv", index=False)
    pd.concat(all_gaps, ignore_index=True).to_csv(REPORT_DIR / "audit_gaps.csv", index=False)
    print("Saved reports/audit_summary.csv and reports/audit_gaps.csv")


if __name__ == "__main__":
    main()