"""Loading utilities for the PRECON dataset."""
from pathlib import Path
import pandas as pd

TIME_COL = "Date_Time"
TOTAL_COL = "Usage_kW"


def house_number(path: Path) -> int:
    """'House12.csv' -> 12"""
    return int("".join(ch for ch in Path(path).stem if ch.isdigit()))


def load_house(path) -> pd.DataFrame:
    """Load one house file into a time-indexed DataFrame (no cleaning yet)."""
    path = Path(path)
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    df = df.rename(columns={TIME_COL: "timestamp"})
    df = df.sort_values("timestamp").set_index("timestamp")

    # float32 halves memory use, and the precision is plenty for meter data
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    df.insert(0, "house_id", house_number(path))
    return df