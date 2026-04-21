"""
Label utilities for classifier training.

Collapses the 6-regime labels from src/labeller/labeller.py into the 3-class
baseline (Up/Down/Sideways) used for Week 4. The 6-state labels remain the
source of truth on disk; this module is a read-only transform used when
training the baseline classifier.

Mapping:
  Trending Up (0)             -> UP (0)
  Trending Down (1)           -> DOWN (1)
  Ranging Neutral (2)         -> SIDEWAYS (2)
  Distribution (3)            -> SIDEWAYS (2)
  Accumulation (4)            -> SIDEWAYS (2)
  Transition/Breakout (5)     -> SIDEWAYS (2)

Labels are read from data/processed/spy_regime_labels.parquet.
"""

import pandas as pd

UP = 0
DOWN = 1
SIDEWAYS = 2

CLASS_3_NAMES = {UP: "Up", DOWN: "Down", SIDEWAYS: "Sideways"}

_SIX_TO_THREE = {0: UP, 1: DOWN, 2: SIDEWAYS, 3: SIDEWAYS, 4: SIDEWAYS, 5: SIDEWAYS}


def collapse_to_3class(regime_id: pd.Series) -> pd.Series:
    """Collapse 6-regime integer labels into the 3-class baseline.

    Args:
        regime_id: Series of integer regime ids in {0, 1, 2, 3, 4, 5}.

    Returns:
        Series of integer 3-class labels in {0, 1, 2}, same index as input.

    Raises:
        ValueError: if any value in regime_id is outside {0..5}.
    """
    unknown = set(regime_id.dropna().unique()) - set(_SIX_TO_THREE.keys())
    if unknown:
        raise ValueError(f"Unexpected regime ids: {sorted(unknown)}")
    return regime_id.map(_SIX_TO_THREE).astype("int64")
