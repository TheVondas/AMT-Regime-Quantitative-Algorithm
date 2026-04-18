import numpy as np
import pandas as pd

from src.features.fracdiff import _get_weights, find_min_d, frac_diff


def test__get_weights_shape():
    """Verify weight array dimensions."""
    window = 50
    weights = _get_weights(d=0.5, window=window)
    assert weights.shape == (window, 1)
    assert weights[-1] == 1.0


def test_frac_diff_d0():
    """d=0 should conceptually return the original series (with warmup NaNs)."""
    series = pd.Series(np.random.randn(150))
    result = frac_diff(series, d=0.0, window=50)
    pd.testing.assert_series_equal(series[49:], result[49:], check_names=False)


def test_frac_diff_d1():
    """d=1 should be equivalent to standard first differencing."""
    randomwalk = pd.Series(np.cumsum(np.random.randn(150)))
    result = frac_diff(randomwalk, d=1.0, window=10)
    expected = randomwalk.diff(1)
    np.testing.assert_allclose(result[10:], expected[10:], atol=1e-7)


def test_find_min_d_stationary():
    """If series is already stationary, find_min_d should return 0."""
    stationary = pd.Series(np.random.randn(200))
    d = find_min_d(stationary, d_step=0.1)
    assert d == 0.0


def test_find_min_d_non_stationary():
    """For a random walk, find_min_d should return a value > 0."""
    randomwalk = pd.Series(np.cumsum(np.random.randn(200)))
    d = find_min_d(randomwalk, d_step=0.1)
    assert d > 0.0
