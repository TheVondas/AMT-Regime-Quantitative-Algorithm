"""
Time-series features for regime classification.

Computes: time reversal asymmetry statistic at lags 1, 2, 3 (rolling 252-day window).

Time reversal asymmetry tests whether a series looks the same played forwards
vs backwards. The statistic is:

    TRA(lag) = mean( x(t) * x(t-lag) * (x(t) - x(t-lag)) )

  - TRA ≈ 0: time-reversible (linear process, random walk, efficient ranging).
  - TRA > 0: positive asymmetry — small values precede large values
    (momentum ignition, breakout, trend acceleration).
  - TRA < 0: negative asymmetry — large values precede small values
    (mean-reversion, volatility decay, post-crisis settling).

This captures the directionality of the generating process — something not
measured by momentum, trend, volatility, or volume features. Trending regimes
show persistent asymmetry, ranging regimes are approximately reversible,
and transitions produce asymmetry spikes.

Three lags capture this at different time scales:
  - Lag 1: day-to-day asymmetry (immediate momentum/reversion).
  - Lag 2: two-day asymmetry (short-term pattern).
  - Lag 3: three-day asymmetry (slightly longer reaction dynamics).

All features are computed from the daily DataFrame in data/processed/daily.parquet.
"""

import pandas as pd


def compute_time_reversal_asymmetry(
    log_returns: pd.Series, lag: int, window: int = 252
) -> pd.Series:
    """Rolling time reversal asymmetry statistic on log returns.

    Computes TRA(lag) = mean( x(t) * x(t-lag) * (x(t) - x(t-lag)) )
    over a rolling window. This is equivalent to:
    mean( x(t)^2 * x(t-lag) - x(t) * x(t-lag)^2 )

    The formula measures third-order temporal structure — whether the
    series has asymmetric dynamics that would look different if time
    were reversed. Linear Gaussian processes (e.g., AR models) have
    TRA = 0 regardless of lag; nonlinear or regime-switching processes
    produce non-zero TRA.

    Args:
        log_returns: Daily log returns series.
        lag: Number of days to lag (1, 2, or 3 for our use case).
        window: Rolling window size in trading days. Default 252 (~1 year),
            consistent with the stationarity features.

    Returns:
        Rolling TRA statistic as a Series. Values near zero indicate
        time-reversibility; deviations indicate asymmetric dynamics.
    """
    x_t = log_returns
    x_lagged = log_returns.shift(lag)

    # TRA = x(t) * x(t-lag) * (x(t) - x(t-lag))
    pointwise = x_t * x_lagged * (x_t - x_lagged)

    return pointwise.rolling(window=window).mean()


def build_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all time-series features from the daily DataFrame.

    Args:
        df: Daily DataFrame with a 'log_return' column.

    Returns:
        DataFrame with time-series feature columns, same index as input.
        Columns: tra_lag1_252, tra_lag2_252, tra_lag3_252.
    """
    log_returns = df["log_return"]

    features = pd.DataFrame(index=df.index)

    features["tra_lag1_252"] = compute_time_reversal_asymmetry(
        log_returns, lag=1, window=252
    )
    features["tra_lag2_252"] = compute_time_reversal_asymmetry(
        log_returns, lag=2, window=252
    )
    features["tra_lag3_252"] = compute_time_reversal_asymmetry(
        log_returns, lag=3, window=252
    )

    return features
