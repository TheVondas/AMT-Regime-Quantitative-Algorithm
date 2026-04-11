"""
Verification plot: time-series features overlaid on SPY price.

Produces a 3-panel chart:
  1. SPY close price
  2. Time reversal asymmetry at lags 1 and 2 (overlaid)
  3. Time reversal asymmetry at lag 3

Lags 1 and 2 share a panel because they are closely related and
comparing them reveals how asymmetry evolves across short time scales.
Lag 3 gets its own panel for clarity.

Saves to notebooks/timeseries_features.png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.features.timeseries import build_timeseries_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"


def main():
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    features = build_timeseries_features(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print verification stats
    valid = features.dropna()
    print(f"Shape: {features.shape}")
    print(f"Usable rows (after warmup): {len(valid)}")
    print(f"Null counts:\n{features.isna().sum()}\n")
    print(f"Summary statistics:\n{valid.describe()}\n")
    print(f"Sample rows (first 5 valid):\n{valid.head()}\n")
    print(f"Sample rows (last 5):\n{valid.tail()}\n")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: SPY Close
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("SPY Close ($)")
    axes[0].set_title("SPY Price with Time Reversal Asymmetry Features")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: TRA lag 1 and lag 2
    axes[1].plot(
        features.index,
        features["tra_lag1_252"],
        color="#2ca02c",
        linewidth=0.7,
        label="TRA lag 1",
    )
    axes[1].plot(
        features.index,
        features["tra_lag2_252"],
        color="#d62728",
        linewidth=0.7,
        alpha=0.7,
        label="TRA lag 2",
    )
    axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[1].set_ylabel("TRA Statistic")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: TRA lag 3
    axes[2].plot(
        features.index,
        features["tra_lag3_252"],
        color="#9467bd",
        linewidth=0.7,
        label="TRA lag 3",
    )
    axes[2].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[2].set_ylabel("TRA Statistic")
    axes[2].set_xlabel("Date")
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
    out_path = OUTPUT_DIR / "timeseries_features.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved time-series verification plot to {out_path}")


if __name__ == "__main__":
    main()
