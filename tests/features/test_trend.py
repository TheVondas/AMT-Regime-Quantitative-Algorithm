import pandas as pd
import pytest

from src.features.trend import build_trend_features, compute_price_sma_ratio


@pytest.fixture
def ohlcv_data():
    """Standard OHLCV structure."""
    idx = pd.date_range("2025-01-01", periods=250)
    return pd.DataFrame(
        {
            "high": [110.0] * 250,
            "low": [90.0] * 250,
            "close": [100.0] * 250,
        },
        index=idx,
    )


def test_compute_price_sma_ratio_above_one():
    """If price is 110 and SMA is 100, ratio should be 1.1."""
    data = pd.Series([100.0] * 9 + [110.0])
    result = compute_price_sma_ratio(data, period=10)
    assert result.iloc[-1] > 1.0


def test_compute_price_sma_ratio_handles_nan(ohlcv_data):
    """Ensure the first (period-1) rows are NaN due to rolling window."""
    result = compute_price_sma_ratio(ohlcv_data["close"], period=50)
    assert pd.isna(result.iloc[0])
    assert not pd.isna(result.iloc[51])


def test_build_trend_features_alignment(ohlcv_data):
    """Ensure all trend features share the same index as input."""
    features = build_trend_features(ohlcv_data)
    pd.testing.assert_index_equal(features.index, ohlcv_data.index)
    assert "sma_cross_ratio" in features.columns
