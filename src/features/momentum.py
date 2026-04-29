"""
Momentum features for regime classification.

Computes: ROC (4 lookbacks), RSI (14-day), CMO (14-day), MACD (12/26/9).
All features are computed from the daily DataFrame in data/processed/daily.parquet.

Each function takes a Series (typically close prices) and returns a Series
with the same index. The build function assembles all momentum features
into a single DataFrame.
"""

from typing import Callable, List

import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import MACD


def compute_roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of Change — percentage change over a lookback period.

    Formula: (close_today - close_n_days_ago) / close_n_days_ago * 100

    Args:
        close (pd.Series): Daily closing prices.
        period (int): Lookback window in trading days.

    Returns:
        pd.Series: ROC values as a Series (percentage scale).
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    return ROCIndicator(close=close, window=period).roc()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — ratio of recent gains to total moves, scaled 0-100.

    RSI > 70 suggests strong upward momentum, < 30 suggests strong downward.
    Persistent RSI > 60 is characteristic of trending-up regimes.

    Args:
        close (pd.Series): Daily closing prices.
        period (int): Lookback window in trading days (default 14).

    Returns:
        pd.Series: RSI values as a Series (0-100 scale).
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    return RSIIndicator(close=close, window=period).rsi()


def compute_cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator — symmetric momentum measure, scaled -100 to +100.

    Similar to RSI but centred on zero: positive = bullish, negative = bearish.
    Uses sum of gains/losses rather than average, so responds differently
    to clustered vs evenly spread price moves.

    Formula: (sum_gains - sum_losses) / (sum_gains + sum_losses) * 100

    Args:
        close (pd.Series): Daily closing prices.
        period (int): Lookback window in trading days (default 14).

    Returns:
        pd.Series: CMO values as a Series (-100 to +100 scale).
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    sum_gains = gains.rolling(window=period).sum()
    sum_losses = losses.rolling(window=period).sum()

    cmo = (sum_gains - sum_losses).divide(sum_gains + sum_losses) * 100
    return cmo.fillna(0.0)


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD — Moving Average Convergence Divergence.

    In a trending-up regime, macd > 0 and histogram > 0 (accelerating).
    Histogram declining while macd still positive often signals transition
    from trend to distribution/ranging.

    Args:
        close (pd.Series): Daily closing prices.
        fast (int): Fast EMA period (default 12).
        slow (int): Slow EMA period (default 26).
        signal (int): Signal line EMA period (default 9).

    Returns:
        pd.DataFrame: DataFrame with columns:
            - macd: EMA(fast) - EMA(slow) — trend direction and strength
            - macd_signal: EMA(signal) of macd — smoothed trend
            - macd_hist: macd - macd_signal — momentum acceleration/deceleration
    """
    assert (
        0 < fast < slow
    ), f"fast ({fast}) must be a positive integer that is less than slow ({slow})."
    assert 0 < signal, f"signal ({signal}) must be a positive integer."
    indicator = MACD(
        close=close, window_fast=fast, window_slow=slow, window_sign=signal
    )
    return pd.DataFrame(
        {
            "macd": indicator.macd(),
            "macd_signal": indicator.macd_signal(),
            "macd_hist": indicator.macd_diff(),
        }
    )


def _compute_for_periods(
    close: pd.Series,
    period: List[int] | int,
    compute_func: Callable[[pd.Series, int], pd.Series],
    feat_name: str,
) -> pd.DataFrame:
    """Map a calculation function over multiple periods and return a DataFrame.

    Args:
        close (pd.Series): Daily closing prices.
        period (List[int] | int): One or more lookback windows.
        compute_func (Callable): Function that takes (Series, int) and returns Series.
        feat_name (str): Prefix for the output columns (e.g., 'roc').

    Returns:
        pd.DataFrame: Combined features with columns
            named '{feat_name}_{p}' for p in period.
    """
    if isinstance(period, int):
        assert f"{feat_name}_period ({period}) must be a positive integer."
        period = [period]

    computations = pd.DataFrame(index=close.index)
    visited = set()

    for i, p in enumerate(period):
        if p in visited:
            continue
        assert (
            p > 0
        ), f"found negative period ({p}) in {feat_name}_period at index: {i}."
        computations[f"{feat_name}_{p}"] = compute_func(close, p)
        visited.add(p)

    return computations


def build_momentum_features(
    df: pd.DataFrame,
    roc_period: List[int] | int = [21, 63, 126, 252],  # (1M, 3M, 6M, 12M)
    rsi_period: List[int] | int = 14,
    cmo_period: List[int] | int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Build all momentum features from the daily DataFrame.

    Args:
        df (pd.DataFrame): Daily DataFrame with at least a 'close' column.
        roc_period (List[int] | int): Lookbacks for Rate of Change.
        rsi_period (List[int] | int): Lookbacks for Relative Strength Index.
        cmo_period (List[int] | int): Lookbacks for Chande Momentum.
        macd_fast (int): Fast EMA for MACD.
        macd_slow (int): Slow EMA for MACD.
        macd_signal (int): Signal line EMA for MACD.

    Returns:
        pd.DataFrame: Momentum features with dynamic column names:
            - roc_{p} for each p in roc_period
            - rsi_{p} for each p in rsi_period
            - cmo_{p} for each p in cmo_period
            - macd, macd_signal, macd_hist
    """
    assert 0 < macd_fast < macd_slow, (
        f"macd_fast ({macd_fast}) must be a positive "
        f"integer that is less than slow ({macd_slow})."
    )
    assert 0 < macd_signal, f"macd_signal ({macd_signal}) must be a positive integer."
    close = df["close"].copy()

    # Rate of Change for specified periods
    roc_df = _compute_for_periods(close, roc_period, compute_roc, "roc")

    # RSI for specified periods
    rsi_df = _compute_for_periods(close, rsi_period, compute_rsi, "rsi")

    # CMO for specified periods
    cmo_df = _compute_for_periods(
        close=close, period=cmo_period, compute_func=compute_cmo, feat_name="cmo"
    )

    # MACD (macd_fast/macd_slow/macd_signal) — returns 3 columns
    macd_df = compute_macd(
        close=close,
        fast=macd_fast,
        slow=macd_slow,
        signal=macd_signal,
    )
    features = pd.DataFrame(index=df.index)
    features = pd.concat([features, roc_df, rsi_df, cmo_df, macd_df], axis=1)

    return features
