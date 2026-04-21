import pandas as pd
import pytest

from src.classifier.labels import DOWN, SIDEWAYS, UP, collapse_to_3class


def test_collapse_mapping():
    """Every 6-class id maps to the expected 3-class id."""
    series = pd.Series([0, 1, 2, 3, 4, 5])
    result = collapse_to_3class(series)
    assert result.tolist() == [UP, DOWN, SIDEWAYS, SIDEWAYS, SIDEWAYS, SIDEWAYS]


def test_collapse_preserves_index():
    """The collapsed series shares the input's index."""
    idx = pd.date_range("2020-01-01", periods=6)
    series = pd.Series([0, 1, 2, 3, 4, 5], index=idx)
    result = collapse_to_3class(series)
    assert result.index.equals(idx)


def test_collapse_rejects_unknown_ids():
    """Unexpected regime ids raise rather than being silently passed through."""
    series = pd.Series([0, 1, 7])
    with pytest.raises(ValueError, match="Unexpected regime ids"):
        collapse_to_3class(series)


def test_collapse_on_real_labels():
    """End-to-end: real labels collapse without loss and sum matches source."""
    labels = pd.read_parquet("data/processed/spy_regime_labels.parquet")
    collapsed = collapse_to_3class(labels["regime_id"])
    assert len(collapsed) == len(labels)
    # 3-class Up count must equal 6-class Trending Up count (label 0)
    assert (collapsed == UP).sum() == (labels["regime_id"] == 0).sum()
    # 3-class Down count must equal 6-class Trending Down count (label 1)
    assert (collapsed == DOWN).sum() == (labels["regime_id"] == 1).sum()
    # 3-class Sideways = sum of labels 2, 3, 4, 5
    assert (collapsed == SIDEWAYS).sum() == labels["regime_id"].isin([2, 3, 4, 5]).sum()
