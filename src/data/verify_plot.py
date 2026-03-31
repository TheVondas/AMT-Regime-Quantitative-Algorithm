"""
Verification plot: SPY price, VIX, and US 10Y yield on the same time axis.

Visual check for data quality and alignment.
Saves plot to notebooks/verification_plot.png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"


def main():
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: SPY Close
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("SPY Close ($)")
    axes[0].set_title("SPY Price, VIX, and US 10Y Yield — Data Verification")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: VIX
    axes[1].plot(df.index, df["vix"], color="#d62728", linewidth=0.7)
    axes[1].set_ylabel("VIX")
    axes[1].axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="VIX = 20")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: US 10Y Yield
    axes[2].plot(df.index, df["us10y"], color="#2ca02c", linewidth=0.7)
    axes[2].set_ylabel("US 10Y Yield (%)")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)

    # Format x-axis
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    # Annotate key events for visual reference
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

    # Mark the train/holdout split
    split_date = pd.Timestamp("2022-01-01")
    for ax in axes:
        ax.axvline(x=split_date, color="purple", linestyle="--", alpha=0.5)
    axes[0].annotate(
        "← Train | Holdout →",
        xy=(split_date, df["close"].max() * 0.95),
        fontsize=8,
        color="purple",
        ha="center",
        xytext=(0, 10),
        textcoords="offset points",
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "verification_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved verification plot to {out_path}")


if __name__ == "__main__":
    main()
