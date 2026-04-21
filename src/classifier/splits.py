"""
Walk-forward cross-validation splitter with purging (PGTS).

Produces (train_idx, val_idx) pairs for training a classifier on expanding
historical windows with a single-year rolling validation period. A purge gap
of 5 trading days is left between the end of the training window and the
start of the validation window to prevent information leakage through
overlapping feature windows and the 1-day lag applied at pipeline assembly.

The holdout period (2022-01-01 onward by default) is never included in any
fold — neither as training data nor as validation. It is reserved for a
single final evaluation at Week 13.

Design decisions (see STATE.md Decision Log 2026-03-31):
  - Training + validation region: 2006-01-09 to 2021-12-31
  - Test holdout: 2022-01-01 to 2024-12-31 (never touched until Week 13)
  - Expanding training window starting from the first available date
  - Validation: non-overlapping 1-calendar-year windows rolling forward
  - Purge: 5 trading days (sufficient given fractional-differencing windows
    are bounded and the 1-day lag is the only look-ahead coupling)

Input is any DataFrame whose index is a DatetimeIndex (e.g. the feature
matrix in data/features/spy_features.parquet). Output positions are integer
indices into that DataFrame, directly usable with sklearn's fit/predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TRAIN_START = pd.Timestamp("2006-01-09")
DEFAULT_HOLDOUT_START = pd.Timestamp("2022-01-01")
DEFAULT_INITIAL_TRAIN_YEARS = 4
DEFAULT_VAL_YEARS = 1
DEFAULT_PURGE_DAYS = 5


@dataclass(frozen=True)
class Fold:
    """A single walk-forward fold.

    Attributes:
        train_idx: Positional indices into the input DataFrame for training.
        val_idx: Positional indices into the input DataFrame for validation.
        train_start: First date in the training window.
        train_end: Last date in the training window (after purge).
        val_start: First date in the validation window.
        val_end: Last date in the validation window.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_start: pd.Timestamp = DEFAULT_TRAIN_START,
    holdout_start: pd.Timestamp = DEFAULT_HOLDOUT_START,
    initial_train_years: int = DEFAULT_INITIAL_TRAIN_YEARS,
    val_years: int = DEFAULT_VAL_YEARS,
    purge_days: int = DEFAULT_PURGE_DAYS,
) -> list[Fold]:
    """Generate walk-forward folds with a 5-day purge between train and val.

    Args:
        index: DatetimeIndex of the dataset being split (e.g. feature matrix).
        train_start: Earliest date to include in any training fold.
        holdout_start: First date of the blocked holdout period — no fold may
            include any data on or after this date.
        initial_train_years: Years of initial training before the first
            validation window.
        val_years: Length of each validation window in years.
        purge_days: Trading days to drop from the tail of each training
            window before the validation window begins.

    Returns:
        Ordered list of Fold records, each yielding positional indices.

    Raises:
        ValueError: if the index is not monotonic, not a DatetimeIndex, or
            contains no data in the requested training region.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("index must be a pandas DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing")
    if holdout_start <= train_start:
        raise ValueError("holdout_start must be after train_start")

    usable = index[(index >= train_start) & (index < holdout_start)]
    if len(usable) == 0:
        raise ValueError(
            f"No data found in [{train_start.date()}, {holdout_start.date()})"
        )

    first_val_start = train_start + pd.DateOffset(years=initial_train_years)
    folds: list[Fold] = []

    val_start = first_val_start
    while val_start < holdout_start:
        val_end = min(val_start + pd.DateOffset(years=val_years), holdout_start)
        val_mask = (index >= val_start) & (index < val_end)
        val_positions = np.flatnonzero(val_mask)
        if len(val_positions) == 0:
            break

        first_val_pos = val_positions[0]
        train_end_pos = first_val_pos - purge_days - 1
        if train_end_pos < 0:
            val_start = val_end
            continue

        train_mask = np.zeros(len(index), dtype=bool)
        train_mask[: train_end_pos + 1] = True
        train_mask &= index >= train_start
        train_positions = np.flatnonzero(train_mask)
        if len(train_positions) == 0:
            val_start = val_end
            continue

        folds.append(
            Fold(
                train_idx=train_positions,
                val_idx=val_positions,
                train_start=index[train_positions[0]],
                train_end=index[train_positions[-1]],
                val_start=index[val_positions[0]],
                val_end=index[val_positions[-1]],
            )
        )
        val_start = val_end

    return folds
