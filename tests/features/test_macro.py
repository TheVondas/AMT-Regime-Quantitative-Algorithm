import pandas as pd
import pytest

from src.features.macro import build_macro_features


@pytest.fixture
def macro_data():
    """Simulate yield curve inversion"""
    return pd.DataFrame(
        {
            "us10y": [4.5] * 30,
            "us5y": [4.0] * 30,
            "us3m": [5.0] * 30,
        },
        index=pd.date_range("2025-01-01", periods=30),
    )


def test_yield_curve_slope_calculation(macro_data):
    """10Y (4.5) - 3M (5.0) should be -0.5."""
    features = build_macro_features(macro_data)
    assert features["yield_curve_10y3m"].iloc[-1] == -0.5
    assert features["yield_curve_10y5y"].iloc[-1] == 0.5


def test_yield_pct_change_logic():
    """Test that a move from 1.0 to 1.1 is recorded as 0.1 (10%)."""
    data = pd.Series([1.0] * 20 + [1.1])
    from src.features.macro import compute_yield_pct_change

    result = compute_yield_pct_change(data, period=20)
    assert pytest.approx(result.iloc[-1]) == 0.1


def test_build_macro_features_passthrough(macro_data):
    """Ensure us10y level is passed through correctly."""
    features = build_macro_features(macro_data)
    assert features["us10y"].iloc[0] == 4.5
