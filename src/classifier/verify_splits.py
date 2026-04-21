"""
Verification plot: walk-forward folds overlaid on the SPY price series.

Produces a single chart with SPY close price as a line and each fold rendered
as two shaded bands (training window in blue, validation window in orange).
The blocked 2022-2024 holdout period is hatched in red to make the discipline
rule visible at a glance.

Used as a manual sanity check that the splitter boundaries match the intended
schedule and that no fold encroaches on the holdout.

Saves to notebooks/walkforward_splits.png.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from src.classifier.splits import DEFAULT_HOLDOUT_START, walk_forward_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "notebooks"

TRAIN_COLOUR = "#1f77b4"
VAL_COLOUR = "#ff7f0e"
HOLDOUT_COLOUR = "#d62728"


def main() -> None:
    features = pd.read_parquet(FEATURES_DIR / "spy_features.parquet")
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    spy = daily["close"].reindex(features.index)

    folds = walk_forward_splits(features.index)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(spy.index, spy.values, color="black", linewidth=0.8, label="SPY close")

    y_min, y_max = ax.get_ylim()
    band_height = (y_max - y_min) * 0.04
    for i, fold in enumerate(folds):
        y_base = y_min + (len(folds) - 1 - i) * band_height
        ax.axvspan(
            fold.train_start,
            fold.train_end,
            ymin=(y_base - y_min) / (y_max - y_min),
            ymax=(y_base + band_height * 0.45 - y_min) / (y_max - y_min),
            color=TRAIN_COLOUR,
            alpha=0.5,
        )
        ax.axvspan(
            fold.val_start,
            fold.val_end,
            ymin=(y_base - y_min) / (y_max - y_min),
            ymax=(y_base + band_height * 0.45 - y_min) / (y_max - y_min),
            color=VAL_COLOUR,
            alpha=0.7,
        )

    ax.axvspan(
        DEFAULT_HOLDOUT_START,
        spy.index.max(),
        color=HOLDOUT_COLOUR,
        alpha=0.15,
        hatch="///",
        label="Holdout (blocked)",
    )

    legend_handles = [
        mpatches.Patch(color=TRAIN_COLOUR, alpha=0.5, label="Training window"),
        mpatches.Patch(color=VAL_COLOUR, alpha=0.7, label="Validation window"),
        mpatches.Patch(
            facecolor=HOLDOUT_COLOUR, alpha=0.15, hatch="///", label="Holdout (blocked)"
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    ax.set_title(
        f"Walk-forward folds ({len(folds)} folds, expanding train + 5-day purge)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("SPY close")
    fig.autofmt_xdate()
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "walkforward_splits.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

    print(f"\nFold summary ({len(folds)} folds):")
    for i, f in enumerate(folds, 1):
        purge = f.val_idx[0] - f.train_idx[-1] - 1
        print(
            f"  fold {i:2d}: train {f.train_start.date()} -> {f.train_end.date()} "
            f"({len(f.train_idx):4d}d), purge={purge}d, "
            f"val {f.val_start.date()} -> {f.val_end.date()} ({len(f.val_idx):3d}d)"
        )


if __name__ == "__main__":
    main()
