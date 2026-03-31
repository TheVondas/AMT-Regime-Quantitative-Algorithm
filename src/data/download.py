"""
Download raw market data from yfinance and save to data/raw/.

Tickers downloaded:
  - SPY:  S&P 500 ETF (OHLCV)
  - ^VIX: CBOE Volatility Index (close)
  - ^TNX: US 10-Year Treasury Yield (close)
  - ^IRX: US 13-Week T-Bill Rate (close, risk-free rate proxy)
  - ^FVX: US 5-Year Treasury Yield (close)

Date range: 2005-01-01 to present.
Output: one Parquet file per ticker in data/raw/.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Configuration ────────────────────────────────────────────────────────────

START_DATE = "2005-01-01"

TICKERS = {
    "SPY": "spy",
    "^VIX": "vix",
    "^TNX": "us10y",
    "^IRX": "us3m",
    "^FVX": "us5y",
}

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ── Download logic ───────────────────────────────────────────────────────────


def download_ticker(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    """Download daily data for a single ticker from yfinance."""
    print(f"  Downloading {ticker} ...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol.")

    # yfinance sometimes returns MultiIndex columns when downloading
    # a single ticker — flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")

    # Ensure index is a clean DatetimeIndex named 'date'
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for ticker, filename in TICKERS.items():
        df = download_ticker(ticker)

        out_path = RAW_DIR / f"{filename}.parquet"
        df.to_parquet(out_path, engine="pyarrow")

        print(
            f"    → Saved {out_path.name}  "
            f"({len(df)} rows, {df.index.min().date()} to {df.index.max().date()})"
        )

    print("\nDone. All raw data saved to", RAW_DIR)


if __name__ == "__main__":
    main()
