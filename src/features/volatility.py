"""
Volatility features for regime classification.

Computes: ATR (14-day, 30-day), rolling standard deviation of returns (20-day,
60-day), VIX level (passthrough), VIX 5-day change.

ATR and rolling std capture realised volatility (what actually happened).
VIX captures implied volatility (what the options market expects).
The gap between realised and implied is itself informative for regime detection.

All features are computed from the daily DataFrame in data/processed/daily.parquet.
"""

from typing import Callable, List

import pandas as pd
from ta.volatility import AverageTrueRange


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range — smoothed average daily price range including gaps.

    True Range is the largest of:
      1. High - Low (today's range)
      2. |High - Previous Close| (gap up)
      3. |Low - Previous Close| (gap down)

    ATR smooths this over the lookback period. Higher ATR = more volatile.
    Trending regimes have moderate steady ATR, ranging regimes have low ATR,
    and breakout/transition regimes show rapidly expanding ATR.

    Args:
        high (pd.Series): Daily high prices.
        low (pd.Series): Daily low prices.
        close (pd.Series): Daily closing prices.
        period (int): Lookback window in trading days (default 14).

    Returns:
        pd.Series: ATR values, expressed as percentage of close price
        for comparability across the full price history.
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    atr = AverageTrueRange(high=high, low=low, close=close, window=period)
    # Express as percentage of close so ATR is comparable
    # across 20 years ($80 SPY in 2005 vs $650 in 2026)
    return atr.average_true_range() / close * 100


def compute_rolling_std(log_returns: pd.Series, period: int) -> pd.Series:
    """Rolling standard deviation of log returns (realised volatility).

    Measures how much close-to-close returns vary over the lookback window.
    Different from ATR: a market that swings intraday but closes flat has
    high ATR but low rolling std.

    Args:
        log_returns (pd.Series): Daily log returns series.
        period (int): Rolling window in trading days.

    Returns:
        pd.Series: Rolling standard deviation as a Series (daily scale, not annualised).
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    return log_returns.rolling(window=period).std()


def compute_vix_change(vix: pd.Series, period: int = 5) -> pd.Series:
    """VIX absolute change over a lookback period.

    Captures the speed and direction of implied volatility shifts.
    A VIX at 25 that was 15 five days ago (spike of +10) signals a very
    different regime state than a stable VIX at 25.

    Rapid VIX spikes typically coincide with transitions out of trending-up
    or ranging regimes into trending-down or breakout.

    Args:
        vix (pd.Series): Daily VIX closing values.
        period (int): Lookback period in trading days.

    Returns:
        pd.Series: Absolute change in VIX over the period.
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    return vix.diff(period)


def compute_vix_pct_change(vix: pd.Series, period: int = 5) -> pd.Series:
    """VIX percentage change over a lookback period.

    Percentage change captures the relative magnitude of VIX moves.
    A +5 point move from VIX 12 (42% spike) is a very different signal
    than a +5 point move from VIX 40 (12.5% rise). Percentage change
    normalises for this, making it more informative for regime detection.

    Args:
        vix (pd.Series): Daily VIX closing values.
        period (int): Lookback period in trading days.

    Returns:
        pd.Series: Percentage change in VIX over the period
            (decimal, not multiplied by 100).
    """
    assert period > 0, f"period ({period}) must be a positive integer."
    return vix.pct_change(period)


def _compute_for_periods(
    period: List[int] | int,
    compute_func: Callable[[int], pd.Series],
    feat_name: str,
    suffix: str = "",
    index: pd.Index = None,
) -> pd.DataFrame:
    """Helper to map a compute function over multiple lookback periods.

    Args:
        periods (List[int] | int): Lookback window(s).
        compute_func (Callable): Function taking (int period) -> Series.
        feat_name (str): Prefix for the column name.
        suffix (str): Suffix to append after the period (e.g., '_pct').
        index (pd.Index): Index for the resulting DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns named '{feat_name}_{p}{suffix}'.
    """
    if isinstance(period, int):
        assert period > 0, f"{feat_name}_period ({period}) must be a positive integer."
        period = [period]

    computations = pd.DataFrame(index=index)
    visited = set()

    for p in period:
        if p in visited:
            continue
        assert p > 0, f"Period must be positive. Found {p} for {feat_name}_period."
        col_name = f"{feat_name}_{p}{suffix}"
        computations[col_name] = compute_func(p)
        visited.add(p)

    return computations


def build_volatility_features(
    df: pd.DataFrame,
    atr_periods: List[int] | int = [14, 30],
    rolling_std_periods: List[int] | int = [20, 60],
    vix_change_periods: List[int] | int = 5,
) -> pd.DataFrame:
    """Build all volatility features from the daily DataFrame.

    Args:
        df (pd.DataFrame): Daily DataFrame with 'high', 'low', 'close',
            'log_return', and 'vix'.
        atr_periods (List[int] | int): Lookbacks for ATR.
        rolling_std_periods (List[int] | int): Lookbacks for Realized Vol.
        vix_change_periods (List[int] | int): Lookbacks for VIX changes.

    Returns:
        pd.DataFrame: Volatility features with dynamic column names:
            - atr_{p}_pct
            - rolling_std_{p}
            - vix (passthrough)
            - vix_change_{p}d
            - vix_pct_change_{p}d
    """
    high, low, close = df["high"].copy(), df["low"].copy(), df["close"].copy()
    log_returns, vix = df["log_return"].copy(), df["vix"].copy()

    features = pd.DataFrame(index=df.index)

    # ATR as percentage of close (14-day and 30-day)
    atr_df = _compute_for_periods(
        period=atr_periods,
        compute_func=lambda p: compute_atr(high, low, close, p),
        feat_name="atr",
        suffix="_pct",
        index=df.index,
    )

    # Rolling standard deviation of log returns
    std_df = _compute_for_periods(
        period=rolling_std_periods,
        compute_func=lambda p: compute_rolling_std(log_returns, p),
        feat_name="rolling_std",
        index=df.index,
    )

    # VIX Passthrough
    features["vix"] = vix

    # VIX Changes
    vix_chg_df = _compute_for_periods(
        period=vix_change_periods,
        compute_func=lambda p: compute_vix_change(vix, p),
        feat_name="vix_change",
        suffix="d",
        index=df.index,
    )
    vix_pct_df = _compute_for_periods(
        period=vix_change_periods,
        compute_func=lambda p: compute_vix_pct_change(vix, p),
        feat_name="vix_pct_change",
        suffix="d",
        index=df.index,
    )

    return pd.concat([features, atr_df, std_df, vix_chg_df, vix_pct_df], axis=1)
