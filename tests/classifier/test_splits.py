import numpy as np
import pandas as pd
import pytest

from src.classifier.splits import (
    DEFAULT_HOLDOUT_START,
    DEFAULT_PURGE_DAYS,
    walk_forward_splits,
)


def _daily_index(start: str, end: str) -> pd.DatetimeIndex:
    """Business-day DatetimeIndex, a reasonable proxy for the feature matrix."""
    return pd.bdate_range(start=start, end=end)


def test_no_train_val_overlap():
    idx = _daily_index("2006-01-02", "2021-12-31")
    for fold in walk_forward_splits(idx):
        assert np.intersect1d(fold.train_idx, fold.val_idx).size == 0


def test_purge_gap_enforced():
    idx = _daily_index("2006-01-02", "2021-12-31")
    for fold in walk_forward_splits(idx):
        gap = fold.val_idx[0] - fold.train_idx[-1] - 1
        assert gap == DEFAULT_PURGE_DAYS


def test_no_fold_leaks_into_holdout():
    idx = _daily_index("2006-01-02", "2024-12-31")
    for fold in walk_forward_splits(idx):
        last_val_date = idx[fold.val_idx[-1]]
        last_train_date = idx[fold.train_idx[-1]]
        assert last_val_date < DEFAULT_HOLDOUT_START
        assert last_train_date < DEFAULT_HOLDOUT_START


def test_training_is_expanding():
    idx = _daily_index("2006-01-02", "2021-12-31")
    folds = walk_forward_splits(idx)
    for earlier, later in zip(folds, folds[1:]):
        assert len(later.train_idx) > len(earlier.train_idx)
        assert later.train_idx[0] == earlier.train_idx[0]


def test_validation_windows_are_disjoint_and_ordered():
    idx = _daily_index("2006-01-02", "2021-12-31")
    folds = walk_forward_splits(idx)
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.val_end < later.val_start


def test_rejects_non_datetime_index():
    idx = pd.Index(range(100))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        walk_forward_splits(idx)


def test_rejects_non_monotonic_index():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-01", "2020-01-03"])
    with pytest.raises(ValueError, match="monotonic"):
        walk_forward_splits(idx)


def test_default_produces_12_yearly_folds():
    idx = _daily_index("2006-01-02", "2024-12-31")
    folds = walk_forward_splits(idx)
    assert len(folds) == 12
