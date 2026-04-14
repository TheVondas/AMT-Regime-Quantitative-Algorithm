"""
Feature pipeline assembly — combines all feature categories into a single matrix.

Steps:
  1. Load daily.parquet
  2. Compute features from all 7 categories
  3. Concatenate and deduplicate (VIX, us10y appear in multiple places)
  4. Test each feature for stationarity (ADF at 5%)
  5. Apply fractional differencing to non-stationary features
  6. Lag all features by 1 day (prevents look-ahead bias)
  7. Drop warmup rows (first 252 days)
  8. Check correlations — flag any pair > 0.95
  9. Save to data/features/spy_features.parquet
  10. Save optimal d values to configs/fracdiff_d_values.json

All features at time t use data up to t-1 only after lagging.
"""

import json
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.features.fracdiff import find_min_d, frac_diff
from src.features.macro import build_macro_features
from src.features.momentum import build_momentum_features
from src.features.stationarity import build_stationarity_features
from src.features.timeseries import build_timeseries_features
from src.features.trend import build_trend_features
from src.features.volatility import build_volatility_features
from src.features.volume import build_volume_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
CONFIGS_DIR = PROJECT_ROOT / "configs"

WARMUP_DAYS = 252
ADF_SIGNIFICANCE = 0.05
CORRELATION_THRESHOLD = 0.95
FRACDIFF_WINDOW = 100


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and concatenate all feature categories, deduplicating overlaps.

    Args:
        df: Daily DataFrame from data/processed/daily.parquet.

    Returns:
        Combined feature DataFrame with unique columns.
    """
    momentum = build_momentum_features(df)
    trend = build_trend_features(df)
    volatility = build_volatility_features(df)
    volume = build_volume_features(df)
    stationarity = build_stationarity_features(df)
    timeseries = build_timeseries_features(df)
    macro = build_macro_features(df)

    features = pd.concat(
        [momentum, trend, volatility, volume, stationarity, timeseries, macro],
        axis=1,
    )

    # Deduplicate: VIX appears in volatility (from build_volatility_features)
    # and us10y appears in macro (from build_macro_features).
    # Both are passthroughs from daily.parquet. Keep one copy of each.
    duplicate_cols = features.columns[features.columns.duplicated()]
    if len(duplicate_cols) > 0:
        print(f"  Removing duplicate columns: {list(duplicate_cols)}")
        features = features.loc[:, ~features.columns.duplicated()]

    return features


def test_stationarity(features: pd.DataFrame) -> dict:
    """Run ADF test on each feature column after dropping warmup NaNs.

    Args:
        features: Feature DataFrame.

    Returns:
        Dict mapping column name to (adf_stat, p_value, is_stationary).
    """
    results = {}
    for col in features.columns:
        clean = features[col].dropna()
        if len(clean) < 50:
            results[col] = (None, None, True)
            continue
        stat, pvalue, *_ = adfuller(clean.values, regression="c", autolag="AIC")
        results[col] = (round(stat, 4), round(pvalue, 6), pvalue < ADF_SIGNIFICANCE)
    return results


def apply_fractional_differencing(
    features: pd.DataFrame, stationarity_results: dict
) -> tuple[pd.DataFrame, dict]:
    """Apply fractional differencing to non-stationary features.

    Args:
        features: Feature DataFrame.
        stationarity_results: Output from test_stationarity().

    Returns:
        Tuple of (differenced DataFrame, dict of d values per feature).
    """
    d_values = {}
    result = features.copy()

    non_stationary = [
        col for col, (_, _, is_stat) in stationarity_results.items() if not is_stat
    ]

    if not non_stationary:
        print("  All features are already stationary — no differencing needed.")
        return result, d_values

    print(f"  Non-stationary features requiring differencing: {non_stationary}")

    for col in non_stationary:
        d = find_min_d(
            features[col],
            significance=ADF_SIGNIFICANCE,
            window=FRACDIFF_WINDOW,
        )
        d_values[col] = d
        print(f"    {col}: optimal d = {d}")

        if d > 0:
            result[col] = frac_diff(features[col], d, window=FRACDIFF_WINDOW)

    return result, d_values


def check_correlations(features: pd.DataFrame) -> list:
    """Find feature pairs with correlation above threshold.

    Args:
        features: Feature DataFrame (NaNs already dropped).

    Returns:
        List of (col1, col2, correlation) tuples exceeding threshold.
    """
    corr = features.corr().abs()
    high_corr = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > CORRELATION_THRESHOLD:
                high_corr.append(
                    (corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 4))
                )

    return high_corr


def main():
    print("Loading daily data...")
    df = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    print("\nBuilding features from all 7 categories...")
    features = build_all_features(df)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Columns ({len(features.columns)}): {list(features.columns)}")

    print("\nTesting stationarity (ADF at 5% significance)...")
    stationarity_results = test_stationarity(features)
    for col, (stat, pvalue, is_stat) in stationarity_results.items():
        status = "STATIONARY" if is_stat else "NON-STATIONARY"
        if stat is not None:
            print(f"  {col:30s}  ADF={stat:8.4f}  p={pvalue:.6f}  {status}")
        else:
            print(f"  {col:30s}  (insufficient data)  {status}")

    print("\nApplying fractional differencing to non-stationary features...")
    features, d_values = apply_fractional_differencing(features, stationarity_results)

    print("\nLagging all features by 1 day (prevent look-ahead bias)...")
    features = features.shift(1)

    print(f"\nDropping warmup period (first {WARMUP_DAYS} rows)...")
    features = features.iloc[WARMUP_DAYS:]

    print("Dropping remaining NaN rows...")
    before = len(features)
    features = features.dropna()
    dropped = before - len(features)
    print(f"  Dropped {dropped} additional rows with NaN")
    print(f"  Final shape: {features.shape}")
    print(
        f"  Date range: {features.index.min().date()} → {features.index.max().date()}"
    )

    print("\nChecking correlations (threshold > 0.95)...")
    high_corr = check_correlations(features)
    if high_corr:
        print("  Highly correlated pairs found:")
        for col1, col2, corr in high_corr:
            print(f"    {col1} <-> {col2}: {corr}")
        print("  NOTE: These are flagged for review. BorutaSHAP will handle")
        print("  redundancy at the feature selection stage (Week 5).")
    else:
        print("  No feature pairs exceed 0.95 correlation.")

    print("\nSummary statistics:")
    print(features.describe().to_string())

    # Save feature matrix
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "spy_features.parquet"
    features.to_parquet(out_path, engine="pyarrow")
    print(f"\nSaved feature matrix to {out_path}")

    # Save d values config
    if d_values:
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        config_path = CONFIGS_DIR / "fracdiff_d_values.json"
        with open(config_path, "w") as f:
            json.dump(d_values, f, indent=2)
        print(f"Saved fractional differencing d values to {config_path}")
    else:
        print("No fractional differencing applied — no d values to save.")

    # Final assertions
    assert features.isna().sum().sum() == 0, "Feature matrix contains NaN values!"
    assert len(features) > 0, "Feature matrix is empty!"
    print(
        f"\nPipeline assembly complete. {len(features.columns)} features, "
        f"{len(features)} rows, zero NaNs."
    )


if __name__ == "__main__":
    main()
