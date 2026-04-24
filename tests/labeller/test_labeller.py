import numpy as np
import pandas as pd
import pytest

from src.labeller.labeller import (
    DISTRIBUTION,
    add_prior_context,
    apply_min_duration,
    compute_kama,
    detect_trend,
    detect_volatility,
)


@pytest.fixture
def ohlcv_data():
    """Generate 200 days of synthetic OHLCV data."""
    np.random.seed(1)
    dates = pd.date_range("2020-01-01", periods=200)
    close = np.linspace(100, 150, 200) + np.random.randn(200)
    df = pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
        },
        index=dates,
    )
    return df


def test_compute_kama_calculation(ohlcv_data):
    """Verify KAMA produces values and respects the warmup period."""
    n = 10
    kama = compute_kama(ohlcv_data["close"], n=n)
    assert len(kama) == len(ohlcv_data)
    assert pd.isna(kama.iloc[n - 1])
    assert not pd.isna(kama.iloc[n])


def test_detect_trend_uptrend():
    """Verify a clear price > KAMA with slope identifies as uptrend."""
    close = pd.Series([100, 105, 110, 115, 120, 125])
    kama = pd.Series([100, 101, 102, 103, 104, 105])  # lagging
    trend = detect_trend(close, kama, slope_window=2, dead_zone_pct=0.01)
    assert trend.iloc[-1] == 1


def test_detect_volatility_detection():
    """Verify high ATR vs SMA(ATR) triggers high vol flag."""

    # low->high("spike") volatility
    high = pd.Series([102] * 50 + [110] * 10)
    low = pd.Series([100] * 50 + [100] * 10)
    close = pd.Series([101] * 50 + [105] * 10)

    vol = detect_volatility(
        high=high, low=low, close=close, atr_period=5, avg_period=20
    )
    assert vol.iloc[10] == 0
    assert vol.iloc[-1] == 1


def test_apply_min_duration_smoothing():
    """Verify that a single period outlier is absorbed into surrounding labels."""
    labels = pd.Series([0, 0, 0, 1, 0, 0, 0])  # labels[3] is single period outlier
    smoothed = apply_min_duration(labels=labels, min_days=3)
    assert smoothed.iloc[3] == 0
    assert (smoothed == 0).all()


def test_add_prior_context():
    """Verify ranging after uptrend is classified as distribution (3)."""
    trend = pd.Series([1] * 25 + [0] * 5)
    basestates = pd.Series([0] * 25 + [2] * 5)  # uptrend then ranging neutral
    refined = add_prior_context(base_states=basestates, trend=trend, prior_window=10)
    assert refined.iloc[-1] == DISTRIBUTION
