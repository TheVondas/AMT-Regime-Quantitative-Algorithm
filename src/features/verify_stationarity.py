"""
Verification plot: stationarity features overlaid on SPY price.

Produces a 3-panel chart:
  1. SPY close price
  2. Rolling ADF test statistic (252-day) with critical value lines
  3. Rolling ADF p-value (252-day) with significance threshold

Saves to notebooks/stationarity_features.png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.features.stationarity import build_stationarity_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"


def main():
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    features = build_stationarity_features(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print verification stats
    valid = features.dropna()
    print(f"Shape: {features.shape}")
    print(f"Usable rows (after 252-day warmup): {len(valid)}")
    print(f"Null counts:\n{features.isna().sum()}\n")
    print(f"Summary statistics:\n{valid.describe()}\n")
    print(f"Sample rows (first 5 valid):\n{valid.head()}\n")
    print(f"Sample rows (last 5):\n{valid.tail()}\n")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: SPY Close
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("SPY Close ($)")
    axes[0].set_title("SPY Price with Rolling ADF Stationarity Features")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: ADF Test Statistic
    axes[1].plot(
        features.index,
        features["adf_stat_252"],
        color="#2ca02c",
        linewidth=0.7,
    )
    # ADF critical values at 1%, 5%, 10% significance (approximate for n=252)
    axes[1].axhline(
        y=-3.46, color="red", linestyle="--", alpha=0.5, label="1% critical (-3.46)"
    )
    axes[1].axhline(
        y=-2.87, color="orange", linestyle="--", alpha=0.5, label="5% critical (-2.87)"
    )
    axes[1].axhline(
        y=-2.57, color="gray", linestyle="--", alpha=0.5, label="10% critical (-2.57)"
    )
    axes[1].set_ylabel("ADF Statistic")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: ADF p-value
    axes[2].plot(
        features.index,
        features["adf_pvalue_252"],
        color="#d62728",
        linewidth=0.7,
    )
    axes[2].axhline(
        y=0.05, color="orange", linestyle="--", alpha=0.5, label="5% significance"
    )
    axes[2].axhline(
        y=0.01, color="red", linestyle="--", alpha=0.5, label="1% significance"
    )
    axes[2].set_ylabel("ADF p-value")
    axes[2].set_xlabel("Date")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].grid(True, alpha=0.3)

    # Format x-axis
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
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
    out_path = OUTPUT_DIR / "stationarity_features.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved stationarity verification plot to {out_path}")


if __name__ == "__main__":
    main()
