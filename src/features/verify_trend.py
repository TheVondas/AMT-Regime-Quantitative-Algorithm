"""
Verification plot: trend features overlaid on SPY price.

Produces a 4-panel chart:
  1. SPY close with SMA(50) and SMA(200)
  2. ADX (14-day) with 20/25 threshold lines
  3. +DI and -DI (14-day)
  4. SMA(50)/SMA(200) crossover ratio

Saves to notebooks/trend_features.png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.features.trend import build_trend_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"


def main():
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    features = build_trend_features(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Panel 1: SPY Close with SMA(50) and SMA(200)
    sma50 = df["close"].rolling(50).mean()
    sma200 = df["close"].rolling(200).mean()
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=0.7, label="SPY")
    axes[0].plot(
        df.index, sma50, color="#ff7f0e", linewidth=0.7, alpha=0.7, label="SMA(50)"
    )
    axes[0].plot(
        df.index, sma200, color="#d62728", linewidth=0.7, alpha=0.7, label="SMA(200)"
    )
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title("SPY Price with Trend Indicators")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: ADX (14-day)
    axes[1].plot(features.index, features["adx_14"], color="#2ca02c", linewidth=0.7)
    axes[1].axhline(
        y=25, color="gray", linestyle="--", alpha=0.5, label="Strong trend (25)"
    )
    axes[1].axhline(
        y=20, color="gray", linestyle=":", alpha=0.4, label="Weak trend (20)"
    )
    axes[1].set_ylabel("ADX (14)")
    axes[1].set_ylim(5, 60)
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: +DI / -DI
    axes[2].plot(
        features.index,
        features["plus_di_14"],
        color="#2ca02c",
        linewidth=0.7,
        label="+DI (buyers)",
    )
    axes[2].plot(
        features.index,
        features["minus_di_14"],
        color="#d62728",
        linewidth=0.7,
        label="-DI (sellers)",
    )
    axes[2].set_ylabel("+DI / -DI (14)")
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: SMA crossover ratio
    axes[3].plot(
        features.index, features["sma_cross_ratio"], color="#9467bd", linewidth=0.7
    )
    axes[3].axhline(
        y=1.0, color="gray", linestyle="--", alpha=0.5, label="Crossover line (1.0)"
    )
    axes[3].set_ylabel("SMA(50) / SMA(200)")
    axes[3].set_xlabel("Date")
    axes[3].legend(loc="upper right", fontsize=7)
    axes[3].grid(True, alpha=0.3)

    # Format x-axis
    axes[3].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    # Mark key events
    events = [
        ("2008-09-15", "GFC"),
        ("2020-03-16", "COVID"),
        ("2022-06-13", "Rate Shock"),
    ]
    for date_str, label in events:
        date = pd.Timestamp(date_str)
        for ax in axes:
            ax.axvline(x=date, color="orange", linestyle=":", alpha=0.6)
        axes[0].annotate(
            label,
            xy=(date, df.loc[date:, "close"].iloc[0]),
            fontsize=7,
            color="orange",
            ha="left",
            xytext=(10, 10),
            textcoords="offset points",
        )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "trend_features.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved trend verification plot to {out_path}")


if __name__ == "__main__":
    main()
