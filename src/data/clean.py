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

# ── Constants & Paths ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MAX_FILL_DAYS = 2

# ── Cleaning logic ───────────────────────────────────────────────────────────


def load_raw(filename: str, raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Load a raw Parquet file and lowercase column names.

    Args:
        filename (str): Name of the parquet file (e.g., 'spy.parquet').
        raw_dir (Path | str): Directory containing raw data files.

    Returns:
        pd.DataFrame: DataFrame with lowercase column names.
    """
    assert filename.endswith(
        ".parquet"
    ), f"filename ({filename}) must have a .parquet suffix"
    file_path = Path(raw_dir) / filename
    assert (file_path).exists(), f"could not find parquet file at path: {file_path}"
    df = pd.read_parquet(file_path)
    df.columns = [c.lower() for c in df.columns]
    return df


def build_daily(
    raw_dir: Path | str = RAW_DIR, max_fill: int = MAX_FILL_DAYS, logging: bool = True
) -> pd.DataFrame:
    """Build the aligned daily DataFrame.

    Args:
        raw_dir (Path | str): Directory containing raw data files.
        max_fill (int): Maximum consecutive days to forward-fill gaps.
        logging (bool): If True, prints status updates to the terminal.

    Returns:
        pd.DataFrame: Aligned and cleaned daily DataFrame.
    """
    assert max_fill >= 0, "max_fill must be a non-negative integer"

    # SPY is the master — its OHLCV forms the base
    spy = load_raw("spy.parquet")

    # Load supplementary series (close only)
    vix = load_raw(filename="vix.parquet", raw_dir=raw_dir)[["close"]].rename(
        columns={"close": "vix"}
    )
    us10y = load_raw(filename="us10y.parquet", raw_dir=raw_dir)[["close"]].rename(
        columns={"close": "us10y"}
    )
    us5y = load_raw(filename="us5y.parquet", raw_dir=raw_dir)[["close"]].rename(
        columns={"close": "us5y"}
    )
    us3m = load_raw(filename="us3m.parquet", raw_dir=raw_dir)[["close"]].rename(
        columns={"close": "us3m"}
    )

    # Join everything onto SPY's index
    df = spy.join([vix, us10y, us5y, us3m], how="left")

    # Forward-fill gaps up to the specified limit (default is 2 for Treasury holidays)
    fill_cols = ["vix", "us10y", "us5y", "us3m"]
    if max_fill != 0:
        df[fill_cols] = df[fill_cols].ffill(limit=max_fill)

    # Compute log return: ln(close_t / close_{t-1})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Drop rows with any remaining NaN (first row due to log_return,
    # plus any start-of-series gaps beyond the specified fill limit)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0 and logging:
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
