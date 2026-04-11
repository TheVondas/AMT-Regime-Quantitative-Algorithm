"""
Stationarity features for regime classification.

Computes: rolling ADF test statistic and p-value (252-day window) on log returns.

The Augmented Dickey-Fuller test measures whether a time series has a unit root
(i.e., is non-stationary). Applied as a rolling window over returns:

  - Strongly negative ADF statistic + low p-value = stationary (mean-reverting,
    characteristic of ranging regimes where price oscillates around a level).
  - Weakly negative / near-zero ADF statistic + high p-value = unit root not
    rejected (trending or random-walk behaviour, characteristic of trending
    regimes where price drifts persistently in one direction).

This gives the classifier a direct statistical signal for "is the market trending
or mean-reverting?" — complementing ADX (which measures trend strength from price
action) with a formal statistical test grounded in time-series econometrics.

Computational note: rolling ADF is expensive (~5 seconds for 5,000+ rows at
252-day window). Compute once; the feature matrix caching handles re-use.

All features are computed from the daily DataFrame in data/processed/daily.parquet.
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def compute_rolling_adf(log_returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """Rolling Augmented Dickey-Fuller test on log returns.

    Applies the ADF test over a sliding window. At each row, the test is
    run on the preceding `window` log returns. The first `window - 1` rows
    will be NaN (insufficient data for the test).

    The ADF test uses a constant (intercept) but no trend term, which is
    appropriate for returns that may have a non-zero mean but should not
    have a deterministic time trend.

    Args:
        log_returns: Daily log returns series.
        window: Rolling window size in trading days. Default 252 (~1 year),
            providing enough data for reliable ADF estimation while still
            capturing regime shifts within the year.

    Returns:
        DataFrame with two columns:
          - adf_stat_252: ADF test statistic (more negative = more stationary).
          - adf_pvalue_252: p-value for the null hypothesis of a unit root
            (lower = stronger evidence of stationarity).
    """
    adf_stats = pd.Series(index=log_returns.index, dtype=float)
    adf_pvalues = pd.Series(index=log_returns.index, dtype=float)

    for i in range(window, len(log_returns) + 1):
        segment = log_returns.iloc[i - window : i]

        # Skip if segment contains NaNs (e.g., early warmup overlap)
        if segment.isna().any():
            continue

        # Skip if segment has zero variance (constant series — ADF undefined)
        if segment.std() == 0:
            continue

        result = adfuller(segment.values, regression="c", autolag="AIC")
        adf_stats.iloc[i - 1] = result[0]
        adf_pvalues.iloc[i - 1] = result[1]

    return pd.DataFrame(
        {
            "adf_stat_252": adf_stats,
            "adf_pvalue_252": adf_pvalues,
        }
    )


def build_stationarity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all stationarity features from the daily DataFrame.

    Args:
        df: Daily DataFrame with a 'log_return' column.

    Returns:
        DataFrame with stationarity feature columns, same index as input.
        Columns: adf_stat_252, adf_pvalue_252.
    """
    return compute_rolling_adf(df["log_return"], window=252)
