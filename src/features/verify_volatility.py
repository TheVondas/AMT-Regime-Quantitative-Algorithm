"""
Verification plot: volatility features overlaid on SPY price.

Produces a 4-panel chart:
  1. SPY close price
  2. ATR as percentage of close (14-day and 30-day)
  3. Rolling standard deviation of returns (20-day and 60-day)
  4. VIX level and VIX 5-day change

Saves to notebooks/volatility_features.png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.features.volatility import build_volatility_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"


def main():
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    features = build_volatility_features(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Panel 1: SPY Close
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("SPY Close ($)")
    axes[0].set_title("SPY Price with Volatility Indicators")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: ATR as % of close (14-day and 30-day)
    axes[1].plot(
        features.index,
        features["atr_14_pct"],
        color="#d62728",
        linewidth=0.7,
        alpha=0.7,
        label="ATR 14d (%)",
    )
    axes[1].plot(
        features.index,
        features["atr_30_pct"],
        color="#ff7f0e",
        linewidth=0.7,
        label="ATR 30d (%)",
    )
    axes[1].set_ylabel("ATR (% of price)")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Rolling standard deviation (20-day and 60-day)
    axes[2].plot(
        features.index,
        features["rolling_std_20"],
        color="#9467bd",
        linewidth=0.7,
        alpha=0.7,
        label="Std 20d",
    )
    axes[2].plot(
        features.index,
        features["rolling_std_60"],
        color="#2ca02c",
        linewidth=0.7,
        label="Std 60d",
    )
    axes[2].set_ylabel("Rolling Std (daily)")
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: VIX level and VIX 5-day change
    ax4a = axes[3]
    ax4b = ax4a.twinx()
    ax4a.plot(
        features.index,
        features["vix"],
        color="#1f77b4",
        linewidth=0.7,
        label="VIX level",
    )
    ax4b.bar(
        features.index,
        features["vix_change_5d"],
        color=features["vix_change_5d"].apply(
            lambda x: "#d62728" if x >= 0 else "#2ca02c"
        ),
        width=1,
        alpha=0.3,
        label="VIX 5d change",
    )
    ax4a.set_ylabel("VIX Level")
    ax4b.set_ylabel("VIX 5d Change")
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
    out_path = OUTPUT_DIR / "volatility_features.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved volatility verification plot to {out_path}")


if __name__ == "__main__":
    main()
