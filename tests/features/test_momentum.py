import numpy as np
import pandas as pd
import pytest

from src.features.momentum import build_momentum_features, compute_cmo


@pytest.fixture
def trend_up_data():
    """Simple upward trend to test momentum direction."""
    return pd.DataFrame(
        {"close": np.linspace(100, 120, 30)},
        index=pd.date_range("2025-01-01", periods=30),
    )


def test_compute_cmo_trending_up_is_positive(trend_up_data):
    """In a steady uptrend, CMO should be positive."""
    result = compute_cmo(trend_up_data["close"], period=14)
    assert result.iloc[-1] > 0
    assert result.iloc[-1] <= 100


def test_compute_cmo_flat_returns_zero():
    """If there is no movement, CMO should be zero after the warmup period."""
    flat = pd.Series([10.0] * 30)
    result = compute_cmo(flat, period=14)

    assert result.iloc[15] == 0.0
    assert result.iloc[-1] == 0.0


def test_build_momentum_features_returns_expected_columns(trend_up_data):
    """Ensure the builder assembles all expected momentum columns."""
    df = build_momentum_features(trend_up_data)
    expected = [
        "roc_21",
        "roc_63",
        "roc_126",
        "roc_252",
        "rsi_14",
        "cmo_14",
        "macd",
        "macd_signal",
        "macd_hist",
    ]
    assert all(col in df.columns for col in expected)
    assert len(df) == len(trend_up_data)
