"""Build the house registry: one row per house, with its quality group and the reason.

Run from the project root:  python -m src.data.registry
"""
from pathlib import Path
import pandas as pd

REPORT_DIR = Path("reports")
MIN_DAYS = 300        # fewer days than this = incomplete annual cycle
MAX_ZERO_PCT = 2.0    # at or above this share of zero readings = unreliable


def assign_group(row):
    if row.days_span < MIN_DAYS:
        return "partial", f"only {row.days_span:.0f} days recorded"
    if row.zero_pct >= MAX_ZERO_PCT:
        return "stress_test", (
            f"{row.zero_pct}% zero readings, longest zero stretch {row.longest_zero_hours} h"
        )
    return "model", ""


def build_registry() -> pd.DataFrame:
    audit = pd.read_csv(REPORT_DIR / "audit_summary.csv", parse_dates=["start", "end"])
    outage = pd.read_csv(REPORT_DIR / "outage_summary.csv")

    keep = ["house_id", "start", "end", "days_span", "mean_kW", "max_kW",
            "repeated_value_pct", "columns"]
    reg = audit[keep].merge(outage, on="house_id")
    reg["frozen_pct"] = (100 * reg.frozen_minutes / (reg.days_span * 1440)).round(2)

    groups = reg.apply(assign_group, axis=1, result_type="expand")
    reg["group"], reg["reason"] = groups[0], groups[1]
    return reg.sort_values("house_id").reset_index(drop=True)


if __name__ == "__main__":
    reg = build_registry()
    reg.to_csv(REPORT_DIR / "house_registry.csv", index=False)
    print(reg.group.value_counts().to_string())
    print()
    print(reg.loc[reg.group != "model", ["house_id", "group", "reason"]].to_string(index=False))