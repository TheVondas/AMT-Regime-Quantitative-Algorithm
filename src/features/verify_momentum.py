"""
Verification plot: momentum features overlaid on SPY price.

Produces a 4-panel chart:
  1. SPY close price
  2. RSI (14-day) with overbought/oversold levels
  3. MACD line, signal line, and histogram
  4. ROC (21-day and 252-day)

Saves to notebooks/momentum_features.png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.features.momentum import build_momentum_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"


def main():
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    features = build_momentum_features(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Panel 1: SPY Close
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("SPY Close ($)")
    axes[0].set_title("SPY Price with Momentum Indicators")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: RSI (14-day)
    axes[1].plot(features.index, features["rsi_14"], color="#d62728", linewidth=0.7)
    axes[1].axhline(
        y=70, color="gray", linestyle="--", alpha=0.5, label="Overbought (70)"
    )
    axes[1].axhline(
        y=30, color="gray", linestyle="--", alpha=0.5, label="Oversold (30)"
    )
    axes[1].axhline(y=50, color="gray", linestyle=":", alpha=0.3)
    axes[1].set_ylabel("RSI (14)")
    axes[1].set_ylim(10, 90)
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: MACD
    axes[2].plot(
        features.index, features["macd"], color="#1f77b4", linewidth=0.7, label="MACD"
    )
    axes[2].plot(
        features.index,
        features["macd_signal"],
        color="#ff7f0e",
        linewidth=0.7,
        label="Signal",
    )
    axes[2].bar(
        features.index,
        features["macd_hist"],
        color=features["macd_hist"].apply(lambda x: "#2ca02c" if x >= 0 else "#d62728"),
        width=1,
        alpha=0.5,
        label="Histogram",
    )
    axes[2].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[2].set_ylabel("MACD (12/26/9)")
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: ROC (21-day and 252-day)
    axes[3].plot(
        features.index,
        features["roc_21"],
        color="#9467bd",
        linewidth=0.7,
        alpha=0.7,
        label="ROC 21d (1M)",
    )
    axes[3].plot(
        features.index,
        features["roc_252"],
        color="#2ca02c",
        linewidth=0.7,
        label="ROC 252d (12M)",
    )
    axes[3].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[3].set_ylabel("ROC (%)")
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
    out_path = OUTPUT_DIR / "momentum_features.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved momentum verification plot to {out_path}")


if __name__ == "__main__":
    main()
