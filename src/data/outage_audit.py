"""Find outage-like (zero) and frozen-sensor stretches in every house.

Run from the project root:  python -m src.data.outage_audit

Assumption: consecutive rows are consecutive minutes. The first audit showed
0 timestamp gaps in every house, so row counts equal minute counts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.loading import load_house, house_number, TOTAL_COL

RAW_DIR = Path("data/raw/precon")
REPORT_DIR = Path("reports")
MIN_ZERO_RUN = 5      # minutes: shorter zero stretches are treated as blips
MIN_FROZEN_RUN = 30   # minutes: identical non-zero readings this long are suspicious


def find_runs(mask: pd.Series) -> pd.DataFrame:
    """Return every stretch of consecutive True values in a boolean Series."""
    m = mask.to_numpy().astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)          # exclusive end position
    idx = mask.index
    return pd.DataFrame({
        "start": idx[starts],
        "end": idx[ends - 1],
        "minutes": ends - starts,
    })


def audit_outages(path: Path):
    df = load_house(path)
    usage = df[TOTAL_COL]
    sub_cols = [c for c in df.columns if c not in ("house_id", TOTAL_COL)]

    # 1. Zero stretches
    zero = usage == 0
    zero_runs = find_runs(zero)
    zero_events = zero_runs[zero_runs.minutes >= MIN_ZERO_RUN].copy()
    zero_events.insert(0, "kind", "zero")

    # 2. Frozen stretches: the same non-zero value repeated
    frozen = (usage.diff() == 0) & (usage != 0)
    frozen_runs = find_runs(frozen)
    frozen_events = frozen_runs[frozen_runs.minutes >= MIN_FROZEN_RUN].copy()
    frozen_events.insert(0, "kind", "frozen")

    events = pd.concat([zero_events, frozen_events], ignore_index=True)
    events.insert(0, "house_id", house_number(path))

    # 3. Days that are (almost) entirely zero
    daily_zero_share = zero.astype("float32").resample("D").mean()

    # 4. When total = 0, is any sub-meter still showing power?
    if sub_cols and zero.any():
        sub_active = (df.loc[zero, sub_cols] > 0.01).any(axis=1)
        sub_active_pct = round(100 * float(sub_active.mean()), 1)
    else:
        sub_active_pct = np.nan

    summary = {
        "house_id": house_number(path),
        "n_submeters": len(sub_cols),
        "zero_minutes": int(zero.sum()),
        "zero_pct": round(100 * float(zero.mean()), 2),
        "zero_events_5min_plus": len(zero_events),
        "longest_zero_hours": round(zero_runs.minutes.max() / 60, 1) if len(zero_runs) else 0.0,
        "fully_zero_days": int((daily_zero_share >= 0.95).sum()),
        "frozen_events_30min_plus": len(frozen_events),
        "frozen_minutes": int(frozen_events.minutes.sum()) if len(frozen_events) else 0,
        "sub_active_when_total_zero_pct": sub_active_pct,
    }
    return summary, events


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    files = sorted(RAW_DIR.glob("House*.csv"), key=house_number)
    print(f"Scanning {len(files)} files...")

    summaries, all_events = [], []
    for i, path in enumerate(files, 1):
        summary, events = audit_outages(path)
        summaries.append(summary)
        all_events.append(events)
        print(f"[{i:2d}/{len(files)}] {path.name}: zero {summary['zero_pct']}%")

    pd.DataFrame(summaries).to_csv(REPORT_DIR / "outage_summary.csv", index=False)
    pd.concat(all_events, ignore_index=True).to_csv(REPORT_DIR / "outage_events.csv", index=False)
    print("Saved reports/outage_summary.csv and reports/outage_events.csv")


if __name__ == "__main__":
    main()