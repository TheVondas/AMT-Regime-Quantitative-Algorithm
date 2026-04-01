"""
Momentum features for regime classification.

Computes: ROC (4 lookbacks), RSI (14-day), CMO (14-day), MACD (12/26/9).
All features are computed from the daily DataFrame in data/processed/daily.parquet.

Each function takes a Series (typically close prices) and returns a Series
with the same index. The build function assembles all momentum features
into a single DataFrame.
"""

import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import MACD


def compute_roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of Change — percentage change over a lookback period.

    Formula: (close_today - close_n_days_ago) / close_n_days_ago * 100

    Args:
        close: Daily closing prices.
        period: Lookback window in trading days.

    Returns:
        ROC values as a Series (percentage scale).
    """
    return ROCIndicator(close=close, window=period).roc()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — ratio of recent gains to total moves, scaled 0-100.

    RSI > 70 suggests strong upward momentum, < 30 suggests strong downward.
    Persistent RSI > 60 is characteristic of trending-up regimes.

    Args:
        close: Daily closing prices.
        period: Lookback window in trading days.

    Returns:
        RSI values as a Series (0-100 scale).
    """
    return RSIIndicator(close=close, window=period).rsi()


def compute_cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator — symmetric momentum measure, scaled -100 to +100.

    Similar to RSI but centred on zero: positive = bullish, negative = bearish.
    Uses sum of gains/losses rather than average, so responds differently
    to clustered vs evenly spread price moves.

    Formula: (sum_gains - sum_losses) / (sum_gains + sum_losses) * 100

    Args:
        close: Daily closing prices.
        period: Lookback window in trading days.

    Returns:
        CMO values as a Series (-100 to +100 scale).
    """
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    sum_gains = gains.rolling(window=period).sum()
    sum_losses = losses.rolling(window=period).sum()

    cmo = (sum_gains - sum_losses) / (sum_gains + sum_losses) * 100
    return cmo


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD — Moving Average Convergence Divergence.

    Returns three components:
      - macd: EMA(fast) - EMA(slow) — trend direction and strength
      - macd_signal: EMA(signal) of macd — smoothed trend
      - macd_hist: macd - macd_signal — momentum acceleration/deceleration

    In a trending-up regime, macd > 0 and histogram > 0 (accelerating).
    Histogram declining while macd still positive often signals transition
    from trend to distribution/ranging.

    Args:
        close: Daily closing prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        DataFrame with columns: macd, macd_signal, macd_hist.
    """
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


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all momentum features from the daily DataFrame.

    Args:
        df: Daily DataFrame with at least a 'close' column.

    Returns:
        DataFrame with momentum feature columns, same index as input.
        Columns: roc_21, roc_63, roc_126, roc_252, rsi_14, cmo_14,
                 macd, macd_signal, macd_hist.
    """
    close = df["close"]

    features = pd.DataFrame(index=df.index)

    # Rate of Change at 4 lookback periods (1M, 3M, 6M, 12M)
    for period in [21, 63, 126, 252]:
        features[f"roc_{period}"] = compute_roc(close, period)

    # RSI (14-day)
    features["rsi_14"] = compute_rsi(close, period=14)

    # CMO (14-day)
    features["cmo_14"] = compute_cmo(close, period=14)

    # MACD (12/26/9) — returns 3 columns
    macd_df = compute_macd(close)
    features = pd.concat([features, macd_df], axis=1)

    return features
