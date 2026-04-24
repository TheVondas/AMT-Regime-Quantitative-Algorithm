import pandas as pd
import pytest

from src.classifier.labels import DOWN, SIDEWAYS, UP, collapse_to_3class


@pytest.fixture
def mock_regime_labels():
    """
    Simulates data/processed/spy_regime_labels.parquet with realistic
    regime distributions and metadata.
    """
    dates = pd.bdate_range(start="2010-01-01", periods=500)
    # Generate a realistic sequence of regimes (0-5)
    rawids = [0] * 100 + [3] * 50 + [1] * 100 + [4] * 50 + [5] * 20 + [2] * 180
    regimenames = {
        0: "Trending Up",
        1: "Trending Down",
        2: "Ranging Neutral",
        3: "Distribution",
        4: "Accumulation",
        5: "Transition/Breakout",
    }
    df = pd.DataFrame(
        {"regime_id": rawids, "regime_label": [regimenames[i] for i in rawids]},
        index=dates,
    )
    return df


def test_collapse_mapping():
    """Every 6-class id maps to the expected 3-class id."""
    series = pd.Series([0, 1, 2, 3, 4, 5])
    result = collapse_to_3class(series)
    assert result.tolist() == [UP, DOWN, SIDEWAYS, SIDEWAYS, SIDEWAYS, SIDEWAYS]


def test_collapse_preserves_index(mock_regime_labels):
    """The collapsed series shares the input's index."""
    result = collapse_to_3class(mock_regime_labels["regime_id"])
    assert result.index.equals(mock_regime_labels.index)


def test_collapse_rejects_unknown_ids():
    """Unexpected regime ids raise rather than being silently passed through."""
    series = pd.Series([0, 1, 7])
    with pytest.raises(ValueError, match="Unexpected regime ids"):
        collapse_to_3class(series)


def test_collapse_logic_on_mock_data(mock_regime_labels):
    """
    Verifies that the collapse logic accurately groups the 6 regimes
    into the 3-class baseline without data loss.
    """
    labels = mock_regime_labels
    collapsed = collapse_to_3class(labels["regime_id"])

    assert len(collapsed) == len(labels)

    # 3-class Up count must equal 6-class Trending Up count (label 0)
    assert (collapsed == UP).sum() == (labels["regime_id"] == 0).sum()

    # 3-class Down count must equal 6-class Trending Down count (label 1)
    assert (collapsed == DOWN).sum() == (labels["regime_id"] == 1).sum()

    # 3-class Sideways = sum of labels 2, 3, 4, 5
    sidewaysmask = labels["regime_id"].isin([2, 3, 4, 5])
    assert (collapsed == SIDEWAYS).sum() == sidewaysmask.sum()
