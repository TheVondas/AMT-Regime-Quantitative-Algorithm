"""
Clean and align raw market data into a single daily DataFrame.

Steps:
  1. Load raw Parquet files from data/raw/
  2. Use SPY trading days as the master date index
  3. Join VIX close, yield closes onto that index
  4. Forward-fill gaps up to 2 days (Treasury holidays)
  5. Compute SPY log returns
  6. Drop any rows with remaining NaNs (start-of-series edge)
  7. Save to data/processed/daily.parquet

Output columns:
  date (index), open, high, low, close, volume,
  vix, us10y, us5y, us3m, log_return
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ── Cleaning logic ───────────────────────────────────────────────────────────


def load_raw(filename: str) -> pd.DataFrame:
    """Load a raw Parquet file and lowercase column names."""
    df = pd.read_parquet(RAW_DIR / filename)
    df.columns = [c.lower() for c in df.columns]
    return df


def build_daily() -> pd.DataFrame:
    """Build the aligned daily DataFrame."""

    # SPY is the master — its OHLCV forms the base
    spy = load_raw("spy.parquet")

    # Load supplementary series (close only)
    vix = load_raw("vix.parquet")[["close"]].rename(columns={"close": "vix"})
    us10y = load_raw("us10y.parquet")[["close"]].rename(columns={"close": "us10y"})
    us5y = load_raw("us5y.parquet")[["close"]].rename(columns={"close": "us5y"})
    us3m = load_raw("us3m.parquet")[["close"]].rename(columns={"close": "us3m"})

    # Join everything onto SPY's index
    df = spy.join([vix, us10y, us5y, us3m], how="left")

    # Forward-fill gaps up to 2 days (Treasury holidays)
    fill_cols = ["vix", "us10y", "us5y", "us3m"]
    df[fill_cols] = df[fill_cols].ffill(limit=2)

    # Compute log return: ln(close_t / close_{t-1})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Drop rows with any remaining NaN (first row due to log_return,
    # plus any start-of-series gaps beyond the 2-day fill limit)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with NaN (start-of-series edge)")

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_daily()
    out_path = PROCESSED_DIR / "daily.parquet"
    df.to_parquet(out_path, engine="pyarrow")

    print(f"\nSaved {out_path}")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Range:   {df.index.min().date()} → {df.index.max().date()}")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nLast 5 rows:")
    print(df.tail().to_string())

    # Summary stats for a quick sanity check
    print("\nSummary statistics:")
    print(df[["close", "vix", "us10y", "us3m", "log_return"]].describe().to_string())


if __name__ == "__main__":
    main()
