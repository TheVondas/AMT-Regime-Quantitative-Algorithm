"""
Verification plot: macro features overlaid on SPY price.

Produces a 4-panel chart:
  1. SPY close price
  2. 10Y yield level with 20-day change (dual axis)
  3. Yield curve slope: 10Y-3M with inversion highlight
  4. Yield curve slope: 10Y-5Y

Saves to notebooks/macro_features.png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.features.macro import build_macro_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"


def main():
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    features = build_macro_features(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print verification stats
    valid = features.dropna()
    print(f"Shape: {features.shape}")
    print(f"Usable rows (after 20-day warmup): {len(valid)}")
    print(f"Null counts:\n{features.isna().sum()}\n")
    print(f"Summary statistics:\n{valid.describe()}\n")
    print(f"Sample rows (first 5 valid):\n{valid.head()}\n")
    print(f"Sample rows (last 5):\n{valid.tail()}\n")

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Panel 1: SPY Close
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("SPY Close ($)")
    axes[0].set_title("SPY Price with Macro Features")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: 10Y yield level + 20-day change (dual axis)
    ax2a = axes[1]
    ax2b = ax2a.twinx()
    ax2a.plot(
        features.index,
        features["us10y"],
        color="#2ca02c",
        linewidth=0.7,
        label="10Y yield (%)",
    )
    ax2b.plot(
        features.index,
        features["us10y_change_20d"],
        color="#d62728",
        linewidth=0.7,
        alpha=0.6,
        label="20d change (pp)",
    )
    ax2b.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2a.set_ylabel("10Y Yield (%)")
    ax2b.set_ylabel("20d Change (pp)")
    ax2a.grid(True, alpha=0.3)
    lines1, labels1 = ax2a.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    # Panel 3: Yield curve slope 10Y-3M with inversion shading
    axes[2].plot(
        features.index,
        features["yield_curve_10y3m"],
        color="#ff7f0e",
        linewidth=0.7,
        label="10Y-3M slope",
    )
    axes[2].axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Inversion line")
    axes[2].fill_between(
        features.index,
        features["yield_curve_10y3m"],
        0,
        where=features["yield_curve_10y3m"] < 0,
        alpha=0.2,
        color="red",
        label="Inverted",
    )
    axes[2].set_ylabel("10Y-3M Slope (pp)")
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Yield curve slope 10Y-5Y
    axes[3].plot(
        features.index,
        features["yield_curve_10y5y"],
        color="#9467bd",
        linewidth=0.7,
        label="10Y-5Y slope",
    )
    axes[3].axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Inversion line")
    axes[3].fill_between(
        features.index,
        features["yield_curve_10y5y"],
        0,
        where=features["yield_curve_10y5y"] < 0,
        alpha=0.2,
        color="red",
        label="Inverted",
    )
    axes[3].set_ylabel("10Y-5Y Slope (pp)")
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
    out_path = OUTPUT_DIR / "macro_features.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved macro verification plot to {out_path}")


if __name__ == "__main__":
    main()
