"""
Volatility features for regime classification.

Computes: ATR (14-day, 30-day), rolling standard deviation of returns (20-day,
60-day), VIX level (passthrough), VIX 5-day change.

ATR and rolling std capture realised volatility (what actually happened).
VIX captures implied volatility (what the options market expects).
The gap between realised and implied is itself informative for regime detection.

All features are computed from the daily DataFrame in data/processed/daily.parquet.
"""

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
        high: Daily high prices.
        low: Daily low prices.
        close: Daily closing prices.
        period: Lookback window in trading days.

    Returns:
        ATR values as a Series, expressed as percentage of close price
        for comparability across the full price history.
    """
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
        log_returns: Daily log returns series.
        period: Rolling window in trading days.

    Returns:
        Rolling standard deviation as a Series (daily scale, not annualised).
    """
    return log_returns.rolling(window=period).std()


def compute_vix_change(vix: pd.Series, period: int = 5) -> pd.Series:
    """VIX absolute change over a lookback period.

    Captures the speed and direction of implied volatility shifts.
    A VIX at 25 that was 15 five days ago (spike of +10) signals a very
    different regime state than a stable VIX at 25.

    Rapid VIX spikes typically coincide with transitions out of trending-up
    or ranging regimes into trending-down or breakout.

    Args:
        vix: Daily VIX closing values.
        period: Lookback period in trading days.

    Returns:
        Absolute change in VIX over the period.
    """
    return vix.diff(period)


def compute_vix_pct_change(vix: pd.Series, period: int = 5) -> pd.Series:
    """VIX percentage change over a lookback period.

    Percentage change captures the relative magnitude of VIX moves.
    A +5 point move from VIX 12 (42% spike) is a very different signal
    than a +5 point move from VIX 40 (12.5% rise). Percentage change
    normalises for this, making it more informative for regime detection.

    Args:
        vix: Daily VIX closing values.
        period: Lookback period in trading days.

    Returns:
        Percentage change in VIX over the period (decimal, not multiplied by 100).
    """
    return vix.pct_change(period)


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all volatility features from the daily DataFrame.

    Args:
        df: Daily DataFrame with 'high', 'low', 'close', 'log_return',
            and 'vix' columns.

    Returns:
        DataFrame with volatility feature columns, same index as input.
        Columns: atr_14_pct, atr_30_pct, rolling_std_20, rolling_std_60,
                 vix, vix_change_5d, vix_pct_change_5d.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    log_returns = df["log_return"]
    vix = df["vix"]

    features = pd.DataFrame(index=df.index)

    # ATR as percentage of close (14-day and 30-day)
    features["atr_14_pct"] = compute_atr(high, low, close, period=14)
    features["atr_30_pct"] = compute_atr(high, low, close, period=30)

    # Rolling standard deviation of log returns (20-day and 60-day)
    features["rolling_std_20"] = compute_rolling_std(log_returns, period=20)
    features["rolling_std_60"] = compute_rolling_std(log_returns, period=60)

    # VIX level (passthrough from daily data)
    features["vix"] = vix

    # VIX 5-day change (absolute and percentage)
    features["vix_change_5d"] = compute_vix_change(vix, period=5)
    features["vix_pct_change_5d"] = compute_vix_pct_change(vix, period=5)

    return features
