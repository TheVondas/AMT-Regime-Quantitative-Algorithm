"""
Verification plot: volume features overlaid on SPY price.

Produces a 4-panel chart:
  1. SPY close price
  2. OBV rate of change (21-day)
  3. MFI (14-day) with overbought/oversold levels
  4. Volume ratio (20-day) and normalised Force Index (13-day)

Saves to notebooks/volume_features.png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.features.volume import build_volume_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"


def main():
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    features = build_volume_features(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Panel 1: SPY Close
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("SPY Close ($)")
    axes[0].set_title("SPY Price with Volume Indicators")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: OBV Rate of Change (21-day)
    axes[1].plot(features.index, features["obv_roc_21"], color="#2ca02c", linewidth=0.7)
    axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[1].set_ylabel("OBV ROC (21d)")
    axes[1].set_ylim(-1, 1)  # Clip view to focus on typical range
    axes[1].grid(True, alpha=0.3)

    # Panel 3: MFI (14-day)
    axes[2].plot(features.index, features["mfi_14"], color="#d62728", linewidth=0.7)
    axes[2].axhline(
        y=80, color="gray", linestyle="--", alpha=0.5, label="Overbought (80)"
    )
    axes[2].axhline(
        y=20, color="gray", linestyle="--", alpha=0.5, label="Oversold (20)"
    )
    axes[2].axhline(y=50, color="gray", linestyle=":", alpha=0.3)
    axes[2].set_ylabel("MFI (14)")
    axes[2].set_ylim(5, 95)
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Volume ratio and Force Index
    ax4a = axes[3]
    ax4b = ax4a.twinx()
    ax4a.plot(
        features.index,
        features["volume_ratio_20"],
        color="#ff7f0e",
        linewidth=0.7,
        label="Vol ratio (20d)",
    )
    ax4a.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
    ax4b.plot(
        features.index,
        features["force_index_norm_13"],
        color="#9467bd",
        linewidth=0.7,
        alpha=0.5,
        label="Force Index norm (13d)",
    )
    ax4b.axhline(y=0, color="gray", linestyle="-", alpha=0.2)
    ax4a.set_ylabel("Volume Ratio")
    ax4b.set_ylabel("Force Index")
    ax4a.set_xlabel("Date")
    ax4a.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax4a.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4a.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

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
    out_path = OUTPUT_DIR / "volume_features.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved volume verification plot to {out_path}")


if __name__ == "__main__":
    main()
