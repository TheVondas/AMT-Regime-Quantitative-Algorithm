"""
Verification plot: regime labels colour-coded on SPY price chart.

Produces a 2-panel chart:
  1. SPY close price with regime-coloured background shading
  2. Regime distribution bar chart (count and percentage per regime)

Also prints regime statistics: count, percentage, and mean duration.

Saves to notebooks/regime_labels.png.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from src.labeller.labeller import (
    ACCUMULATION,
    DISTRIBUTION,
    RANGING_NEUTRAL,
    REGIME_NAMES,
    TRANSITION,
    TRENDING_DOWN,
    TRENDING_UP,
    build_regime_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"

# Colour scheme for regimes
REGIME_COLOURS = {
    TRENDING_UP: "#2ca02c",  # green
    TRENDING_DOWN: "#d62728",  # red
    RANGING_NEUTRAL: "#7f7f7f",  # grey
    DISTRIBUTION: "#ff7f0e",  # orange
    ACCUMULATION: "#1f77b4",  # blue
    TRANSITION: "#9467bd",  # purple
}


def compute_regime_stats(labels: pd.DataFrame) -> pd.DataFrame:
    """Compute regime distribution statistics.

    Args:
        labels: DataFrame with 'regime_id' and 'regime_label' columns.

    Returns:
        DataFrame with count, percentage, and mean duration per regime.
    """
    valid = labels.dropna(subset=["regime_id"])
    total = len(valid)

    stats = []
    for regime_id in sorted(REGIME_NAMES.keys()):
        mask = valid["regime_id"] == regime_id
        count = mask.sum()
        pct = count / total * 100

        # Compute mean duration of contiguous runs
        runs = (mask != mask.shift()).cumsum()
        regime_runs = runs[mask]
        if len(regime_runs) > 0:
            durations = regime_runs.value_counts().values
            mean_dur = durations.mean()
        else:
            mean_dur = 0

        stats.append(
            {
                "regime_id": regime_id,
                "regime_label": REGIME_NAMES[regime_id],
                "count": count,
                "percentage": round(pct, 1),
                "mean_duration_days": round(mean_dur, 1),
            }
        )

    return pd.DataFrame(stats)


def main():
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    labels = build_regime_labels(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print stats
    valid = labels.dropna(subset=["regime_id"])
    print(f"Total labelled days: {len(valid)}")
    print(f"Date range: {valid.index.min().date()} → {valid.index.max().date()}")
    print("\nRegime distribution:")
    stats = compute_regime_stats(labels)
    print(stats.to_string(index=False))
    print("\nSample labels (first 10 valid):")
    print(valid.head(10).to_string())

    # Check minimum sample count
    min_count = stats["count"].min()
    min_regime = stats.loc[stats["count"].idxmin(), "regime_label"]
    if min_count < 50:
        print(
            f"\n⚠ WARNING: {min_regime} has only {min_count} samples "
            f"(< 50 minimum). Consider collapsing."
        )
    else:
        print(
            f"\nAll regimes have ≥ 50 samples (minimum: {min_count} "
            f"for {min_regime})."
        )

    # Plot
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Panel 1: SPY price with regime shading
    ax1 = axes[0]
    ax1.plot(df.index, df["close"], color="black", linewidth=0.5, zorder=3)

    # Add regime background shading
    regime_ids = valid["regime_id"].values
    dates = valid.index

    i = 0
    while i < len(regime_ids):
        regime = regime_ids[i]
        start = dates[i]
        # Find end of this regime run
        j = i
        while j < len(regime_ids) and regime_ids[j] == regime:
            j += 1
        end = dates[j - 1]
        colour = REGIME_COLOURS.get(regime, "#cccccc")
        ax1.axvspan(start, end, alpha=0.2, color=colour, linewidth=0)
        i = j

    ax1.set_ylabel("SPY Close ($)")
    ax1.set_title("SPY Price with Regime Labels")
    ax1.grid(True, alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(color=REGIME_COLOURS[rid], alpha=0.4, label=REGIME_NAMES[rid])
        for rid in sorted(REGIME_NAMES.keys())
    ]
    ax1.legend(handles=patches, loc="upper left", fontsize=7, ncol=2)

    # Mark key events
    events = [
        ("2008-09-15", "GFC"),
        ("2020-03-16", "COVID"),
        ("2022-06-13", "Rate Shock"),
    ]
    for date_str, label in events:
        date = pd.Timestamp(date_str)
        ax1.axvline(x=date, color="orange", linestyle=":", alpha=0.8, zorder=4)
        ax1.annotate(
            label,
            xy=(date, df.loc[date:, "close"].iloc[0]),
            fontsize=7,
            color="orange",
            ha="left",
            xytext=(10, 10),
            textcoords="offset points",
            zorder=5,
        )

    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: Regime distribution bar chart
    ax2 = axes[1]
    colours = [REGIME_COLOURS[rid] for rid in stats["regime_id"]]
    bars = ax2.bar(stats["regime_label"], stats["count"], color=colours, alpha=0.7)

    # Add percentage labels on bars
    for bar, pct in zip(bars, stats["percentage"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{pct}%",
            ha="center",
            fontsize=8,
        )

    ax2.set_ylabel("Number of Days")
    ax2.set_title("Regime Distribution")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "regime_labels.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nSaved regime verification plot to {out_path}")


if __name__ == "__main__":
    main()
