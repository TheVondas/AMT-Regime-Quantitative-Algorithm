# Development Plan — Week-by-Week Execution

This plan assumes part-time effort (evenings/weekends). Each week has concrete deliverables and a clear definition of done. No week depends on anything beyond the prior week's output. If a week takes longer, that's fine — the sequencing matters more than the calendar.

---

## Stage 1: Foundation and Data Pipeline

**Goal:** Get raw data flowing, features computing, and a baseline labeller producing regime labels for one instrument. No ML yet — just plumbing.

---

### Week 1 — Environment Setup and Data Acquisition

**What you're doing:** Setting up the project structure, sourcing daily data, and confirming you can pull clean OHLCV programmatically.

**Tasks:**

- [ ] Set up Python environment (venv or conda) with core dependencies:
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `yfinance`, `optuna`, `shap`, `statsmodels`, `ta-lib` (or `ta`), `fracdiff`, `joblib`
- [ ] Create project directory structure:
  ```
  AMT-Regime-Quantitative-Algorithm/
  ├── data/
  │   ├── raw/              # Downloaded OHLCV, VIX, yields
  │   ├── processed/        # Cleaned, aligned daily data
  │   └── features/         # Computed feature matrices (cached)
  ├── src/
  │   ├── data/             # Data download and cleaning scripts
  │   ├── features/         # Feature engineering modules
  │   ├── labeller/         # Regime labelling logic
  │   ├── classifier/       # ML classifier training and prediction
  │   ├── strategy/         # Per-regime strategy templates
  │   ├── backtest/         # Backtesting engine
  │   └── analysis/         # Alpha measurement, factor regression, SHAP
  ├── notebooks/            # Exploration and visualisation
  ├── tests/                # Unit tests
  └── configs/              # Hyperparameter configs, instrument lists
  ```
- [ ] Write a data download script for SPY daily OHLCV (2005-present) using yfinance
- [ ] Download VIX daily close (CBOE via yfinance: `^VIX`)
- [ ] Download US 10Y yield and US 2Y yield from FRED (use `fredapi` or manual CSV download from https://fred.stlouisfed.org)
- [ ] Download US 3-month T-bill rate from FRED (risk-free rate for Sharpe/Sortino calculations)
- [ ] Write a data cleaning function: handle missing dates, forward-fill gaps of 1-2 days, align all series to the same trading day index, compute log returns
- [ ] Save cleaned data to `data/processed/` as Parquet or CSV
- [ ] Verify: plot SPY price, VIX, and 10Y yield on the same time axis. Visually confirm data quality and alignment

**Definition of done:** You can run one script that downloads, cleans, and saves aligned daily data for SPY + VIX + yields from 2005 to present. Output is a single DataFrame with columns: `date, open, high, low, close, volume, vix, us10y, us2y, us3m, log_return`.

**Knowledge required:** Basic pandas, yfinance API, FRED data access.

**Tools required:** Python, pip/conda, yfinance, internet connection.

---

### Week 2 — Pomorski-Style Feature Engineering

**What you're doing:** Computing the ~20 core features from daily data that Pomorski showed are most predictive. No AMT features yet — those come in Stage 2.

**Tasks:**

- [ ] Implement momentum features:
  - ROC (Rate of Change) at 21, 63, 126, 252 day lookbacks (≈ 1M, 3M, 6M, 12M)
  - RSI (14-day)
  - CMO (Chande Momentum Oscillator, 14-day)
  - MACD (12/26/9)
- [ ] Implement trend features:
  - ADX (14-day)
  - Plus DI / Minus DI (14-day)
  - Price vs SMA(50), price vs SMA(200)
  - SMA(50) vs SMA(200) crossover flag
- [ ] Implement volatility features:
  - ATR (14-day, 30-day)
  - Rolling standard deviation of returns (20-day, 60-day)
  - VIX level (already in data)
  - VIX 5-day change
- [ ] Implement volume features:
  - OBV (On-Balance Volume)
  - Volume ratio: current volume / 20-day average volume
  - MFI (Money Flow Index, 14-day)
  - Force Index (13-day EMA)
- [ ] Implement stationarity features:
  - Rolling ADF test statistic (252-day window) — this is slow, compute once and cache
  - Rolling ADF p-value (252-day window)
- [ ] Implement time-series features:
  - Time reversal asymmetry statistic at lags 1, 2, 3
  - (Optional: energy ratio by chunks from tsfresh — add if time permits)
- [ ] Implement macro features:
  - 10Y yield level, 10Y yield 20-day change
  - Yield curve slope: 10Y - 2Y
- [ ] Apply fractional differencing to non-stationary features:
  - Use `fracdiff` package or implement López de Prado's fixed-width window method
  - For each feature: find minimum d such that the series passes ADF test at 5% significance
  - Store optimal d values in a config file for reproducibility
- [ ] Lag all features by 1 day (feature at time t uses data up to t-1 only)
- [ ] Save complete feature matrix to `data/features/spy_features.parquet`
- [ ] Verify: print feature correlation matrix. Check no feature is >0.95 correlated with another (drop one if so). Check no NaNs in the output after warmup period

**Definition of done:** A feature matrix with ~20-25 columns, one row per trading day, all lagged by 1 day, fractionally differenced where needed, saved to disk. Warmup period (first 252 days) excluded from output.

**Knowledge required:** Technical indicator formulas (or use ta-lib), fractional differencing concept from López de Prado (2018), ADF test interpretation.

**Tools required:** `ta` or `ta-lib`, `fracdiff`, `statsmodels` (for ADF test).

---

### Week 3 — Rule-Based Regime Labeller (v1)

**What you're doing:** Building the ground truth generator. This assigns one of 6 regime labels to every trading day in your history. The classifier will later learn to predict these labels.

**Tasks:**

- [ ] Implement trend direction detection:
  - Compute KAMA (Kaufman's Adaptive Moving Average) on daily close, default parameters: n=10, n_s=2, n_l=30
  - Alternatively (simpler): use a 50-day EMA as starting point
  - Trend up: price > MA and MA slope positive (MA today > MA 5 days ago)
  - Trend down: price < MA and MA slope negative
  - No trend: everything else
- [ ] Implement volatility regime detection:
  - Compute ATR(14)
  - High volatility: ATR(14) > SMA(ATR(14), 50)
  - Low volatility: ATR(14) ≤ SMA(ATR(14), 50)
- [ ] Combine into 4 base states:
  - Trend up + low vol → **Trending Up**
  - Trend down + high vol → **Trending Down**
  - Trend up + high vol → transitory bullish
  - Trend down + low vol → transitory bearish
- [ ] Add prior-state context to produce 6 states:
  - If current state is "no trend" AND prior sustained state was "Trending Up" → **Ranging After Uptrend**
  - If current state is "no trend" AND prior sustained state was "Trending Down" → **Ranging After Downtrend**
  - If current state is "no trend" AND prior sustained state was also "no trend" or mixed → **Ranging Neutral**
  - Define "prior sustained state" as the dominant state over the previous 20 trading days
  - Transitory bullish/bearish states: classify as **Transition/Breakout** if they last < 5 days, otherwise reclassify based on the trend direction
- [ ] Apply minimum regime duration filter:
  - Any regime lasting < 5 days is absorbed into the surrounding regime
  - Use a smoothing pass: forward-fill regimes shorter than 5 days
- [ ] Validate labeller output:
  - Plot SPY price with regime labels colour-coded as background shading
  - Visually check: do the labels make intuitive sense? Does "Trending Up" cover bull runs? Does "Ranging After Uptrend" cover distribution phases?
  - Print regime distribution: count and percentage of days in each regime
  - Check: no regime has fewer than ~50 occurrences (if so, consider collapsing)
- [ ] Sensitivity test:
  - Rerun labeller with MA lookback = 40 and 60 (instead of 50). Compare label distributions. If >20% of days change labels, the labeller is unstable — adjust parameters or approach
- [ ] Save labels to `data/processed/spy_regime_labels.parquet` (columns: `date, regime_label, regime_id`)

**Definition of done:** Every trading day from 2006 onwards (after warmup) has one of 6 regime labels. Visually validated on price chart. No regime has fewer than 50 training samples. Labels are stable under small parameter changes.

**Knowledge required:** KAMA formula (Pomorski Section 2.2.2), ATR, basic understanding of Wyckoff distribution/accumulation phases.

**Tools required:** Just numpy/pandas. No ML libraries needed yet.

---

### Week 4 — Baseline Classifier (3-Class)

**What you're doing:** Training a Random Forest to predict regimes from features. Starting with 3 classes (up/down/sideways) to validate the pipeline before attempting 6 classes.

**Tasks:**

- [ ] Collapse 6 regime labels into 3 for the baseline:
  - Trending Up → **Up**
  - Trending Down → **Down**
  - All ranging + transition states → **Sideways**
- [ ] Set up walk-forward validation:
  - Training window: expanding, starting from 2006
  - Validation window: 1 year, rolling forward
  - Test holdout: 2022-2024 (never touched until final evaluation)
  - Implement PGTS: purge 5 days between train and validation to prevent leakage
- [ ] Train a Random Forest classifier:
  - Initial hyperparameters from Pomorski: `n_estimators=250, max_depth=10, min_samples_leaf=80, max_features=0.3`
  - `class_weight="balanced"` to handle class imbalance
  - Fit on training set, predict on validation set
- [ ] Evaluate classification performance:
  - Compute MCC per class and overall
  - Print confusion matrix: which regimes get confused?
  - Target: MCC > 0.3 per class on validation data
- [ ] Evaluate financial performance (simple backtest):
  - Strategy: long SPY when predicted Up, short (or flat) when predicted Down, flat when predicted Sideways
  - Compute: cumulative returns, Sortino ratio, max drawdown
  - Include transaction costs: 0.2% per trade (round-trip)
  - Compare vs buy-and-hold SPY over same period
- [ ] Run SHAP analysis:
  - Compute SHAP values on validation set
  - Plot top 15 features by importance
  - Check: do momentum features dominate (expected)? Is there anything surprising?
- [ ] Save trained model with `joblib.dump()`

**Definition of done:** RF classifier achieves MCC > 0.3 on 3-class validation set. Simple trading strategy beats buy-and-hold after costs on validation set. SHAP analysis shows sensible feature importances. Model saved to disk.

**Knowledge required:** scikit-learn RandomForestClassifier API, walk-forward validation concept, MCC interpretation, basic SHAP usage.

**Tools required:** `scikit-learn`, `shap`, `matplotlib`.

---

### Week 5 — Expand to 6-Class Classifier and Hyperparameter Tuning

**What you're doing:** Expanding to the full 6-class problem and properly tuning hyperparameters.

**Tasks:**

- [ ] Retrain RF classifier on full 6-class labels using same walk-forward setup
- [ ] Evaluate 6-class MCC:
  - MCC per regime class
  - Confusion matrix: which states are confused with which?
  - If any regime has MCC < 0.15, it's not learnable — collapse it into the nearest state and re-evaluate
- [ ] Hyperparameter tuning with Optuna:
  - Objective: maximise Sortino ratio on walk-forward validation splits (Pomorski's approach)
  - Search space:
    ```
    n_estimators: 100-300
    max_depth: 3-15
    min_samples_split: 10-100
    min_samples_leaf: 30-100
    max_samples: 0.1-0.5
    max_features: 0.15-0.5
    min_weight_fraction_leaf: 0.0-0.05
    ```
  - Run 200 trials. Should complete in 30-60 minutes on a laptop
- [ ] Feature selection with BorutaSHAP:
  - Run BorutaSHAP to identify confirmed/rejected/tentative features
  - Drop rejected features, keep confirmed + tentative
  - Retrain with reduced feature set and compare MCC — should hold or improve slightly
- [ ] Retrain final RF with optimal hyperparameters and selected features
- [ ] Compare 3-class vs 6-class performance:
  - Does the 6-class model produce better financial results than 3-class?
  - If not, the additional states aren't adding value yet — proceed with 3-class and revisit after AMT features are added
- [ ] Save tuned model, optimal hyperparameters, and selected features to configs

**Definition of done:** Tuned 6-class (or confirmed 3-class) RF classifier. Optimal hyperparameters documented. Feature set pruned. Performance baseline established for Stage 1 (Pomorski features only).

**Knowledge required:** Optuna API (simple), BorutaSHAP usage, understanding of RF hyperparameters.

**Tools required:** `optuna`, `BorutaSHAP` (pip install boruta_shap).

---

### Week 6 — Stage 1 Backtest and Alpha Measurement

**What you're doing:** Rigorous backtesting of the Stage 1 system and measuring whether you have alpha. This is the gate — if results are negative here, debug before moving to Stage 2.

**Tasks:**

- [ ] Build a proper backtesting engine:
  - Walk-forward: retrain classifier every 6 months on expanding window
  - At each retraining point: re-optimise hyperparameters (or use fixed from Week 5 for speed)
  - Generate daily regime predictions for the full validation period
  - Apply the simple trading strategy: long/short/flat based on regime prediction
  - Track: daily returns, positions, trade log, equity curve
  - Deduct transaction costs on every trade (0.2% round-trip)
- [ ] Compute financial metrics on the full validation period:
  - Annualised Sortino ratio
  - Annualised Sharpe ratio
  - Cumulative returns vs buy-and-hold
  - Maximum drawdown
  - Win rate
  - Number of trades and annualised turnover
- [ ] Run Fama-French factor regression:
  - Download Fama-French 5 factors + momentum from Kenneth French's website (CSV, free)
  - Align daily factor returns with your strategy's daily returns
  - Run OLS: `R_strategy - R_f = α + β₁(MKT-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA) + β₆(MOM) + ε`
  - Check: is α positive? Is t-stat > 2.0? What is β₆ (momentum loading)?
- [ ] Run attribution tests:
  - Test A: your classifier + your strategy (already done above)
  - Test D: random regime labels + your strategy (run 100 random shuffles, take average). If A is not significantly better than D, the classifier isn't working
  - Test C: your classifier + simple always-long strategy. If A ≈ C, the regime-specific positioning isn't helping
- [ ] Bootstrap confidence interval:
  - Resample daily returns 10,000 times with replacement
  - Compute alpha for each bootstrap sample
  - Report 5th and 95th percentile of alpha distribution
  - If 5th percentile is negative, alpha is not statistically robust
- [ ] Document results: save all metrics, plots, and regression outputs to `notebooks/stage1_results.ipynb`

**Definition of done:** Complete Stage 1 evaluation documented in a notebook. Clear answer to: "Does the Pomorski-features-only, single-instrument system produce statistically significant alpha?" If yes, proceed to Stage 2. If no, diagnose (labeller issue? feature issue? strategy issue?) and iterate on Stage 1 before proceeding.

**Knowledge required:** OLS regression (statsmodels), Fama-French factor model interpretation, bootstrap resampling.

**Tools required:** `statsmodels`, Fama-French data (CSV download), `scipy` (for bootstrap).

---

## Stage 1 Gate

**Before proceeding to Stage 2, you must have:**

- [ ] MCC > 0.25 on the actionable regime classes (up/down at minimum)
- [ ] Positive cumulative returns after costs vs buy-and-hold
- [ ] Strategy returns that are significantly better than random regime labels (Test A >> Test D)
- [ ] Understanding of which features are driving predictions (SHAP)
- [ ] A clean, modular codebase where you can swap components

**If you don't pass the gate:** revisit the labeller (is it producing sensible regimes?), the features (are they leaking future info? are they actually predictive?), or the strategy (is long/short/flat too naive?). Do not add complexity until the simple version works.

---

## Stage 2: AMT Feature Integration

**Goal:** Add market microstructure features from AMT tools and measure whether they improve classifier performance.

---

### Week 7 — Intraday Data Acquisition and AMT Feature Engine (Part 1)

**What you're doing:** Sourcing intraday data and building the infrastructure to compute TPO and Volume Profile from bar data.

**Tasks:**

- [ ] Source intraday bar data for SPY:
  - Option A: Polygon.io free tier (2 years of 5-min bars, sufficient for initial testing)
  - Option B: Alpaca free tier (similar coverage)
  - Option C: yfinance (7 days of 1-min, 60 days of 5-min — very limited, use only for prototyping)
  - If using futures (ES), consider Databento ($50-100/mo for clean tick/bar data)
  - Target: at least 3 years of 30-min bars minimum, 5 years of 5-min bars preferred
- [ ] Write intraday data download and storage pipeline:
  - Download in chunks (API rate limits), save per-day or per-month Parquet files
  - Handle market hours: filter to RTH only (9:30-16:00 ET for SPY)
  - Validate: check for gaps, zero-volume bars, price spikes
- [ ] Implement TPO (Time Price Opportunity) computation:
  - Divide each session into 30-min periods (standard Market Profile)
  - Divide price range into buckets (bucket width = 0.1 × daily ATR for adaptive sizing)
  - For each bucket, count how many 30-min periods traded at that price → TPO count
  - Compute daily TPO-derived features:
    - Value Area High (VAH): upper bound of 70% volume concentration
    - Value Area Low (VAL): lower bound of 70% volume concentration
    - Point of Control (POC): price bucket with highest TPO count
    - Value area width: (VAH - VAL) / close, expressed as percentage
    - TPO count at POC: time concentration measure
    - Distribution shape: classify as P (long above POC), b (long below POC), D (symmetric), or other
- [ ] Implement Volume Profile computation:
  - From intraday bars: sum volume in each price bucket per session
  - Compute:
    - Volume POC (may differ from TPO POC)
    - Volume-based VAH/VAL
    - High Volume Nodes (HVN): price buckets with volume > 1.5× average bucket volume
    - Low Volume Nodes (LVN): price buckets with volume < 0.5× average bucket volume
    - HVN count in current range
    - Nearest LVN above and below current price (distance as ATR multiple)
    - Volume concentration ratio: % of total volume in top 3 price buckets
- [ ] Cache daily AMT summary features to `data/features/spy_amt_daily.parquet`
- [ ] Verify: plot a few days of TPO profiles alongside price. Does the value area make sense? Does POC align with the highest-activity price?

**Definition of done:** Automated pipeline that computes daily TPO and Volume Profile features from intraday bars, saves to disk. Features are ATR-normalised and ready to merge with the daily feature matrix.

**Knowledge required:** Market Profile / TPO construction (Steidlmayer), Volume Profile concept, how to define value area (70% rule).

**Tools required:** Polygon.io or Alpaca API client, pandas for bar aggregation.

---

### Week 8 — AMT Feature Engine (Part 2: VWAP, Delta, OI)

**What you're doing:** Adding the remaining AMT features and merging everything into a unified feature matrix.

**Tasks:**

- [ ] Implement VWAP features:
  - Session VWAP: cumulative (price × volume) / cumulative volume, reset each session
  - Weekly VWAP: rolling 5-session VWAP (do not reset daily)
  - Daily features:
    - Close vs session VWAP (above/below, distance as ATR multiple)
    - Close vs weekly VWAP
    - Session VWAP slope: VWAP at close vs VWAP at midday (directional development)
    - VWAP touch count (5-day): how many sessions closed within 0.25 ATR of VWAP (ranging indicator)
- [ ] Implement Volume Delta features (approximated from bars):
  - Bar-level delta: if close > open, bar volume is positive (buying); if close < open, negative (selling). If close = open, split 50/50
  - Per-session cumulative delta: sum of bar-level deltas
  - Daily features:
    - Cumulative delta (5-day rolling sum)
    - Delta divergence: sign(cumulative_delta_5d) ≠ sign(return_5d) → divergence flag
    - Delta at VAH: sum of delta for bars where price was within 1 bucket of VAH
    - Delta at VAL: sum of delta for bars where price was within 1 bucket of VAL
    - Delta rate of change: today's session delta minus yesterday's
  - Note: bar-level delta is approximate. Acceptable for daily regime classification. True trade-level delta is a future upgrade
- [ ] Implement Open Interest features (if trading futures):
  - Source: CME daily settlement data or broker API
  - Daily features:
    - OI change (1-day)
    - OI vs price divergence: rising price + falling OI = short covering (not new longs), flag
    - OI percentile: current OI as percentile of 30-day OI range
  - If trading SPY (equity), skip OI — not applicable. Add later if expanding to futures
- [ ] Implement derived AMT features:
  - VA migration (1-day): today's VA midpoint vs yesterday's VA midpoint, as ATR multiple
  - VA migration (5-day): VA midpoint 5-day change
  - POC migration rate: 5-day trend of POC position
  - Single prints above VA: count of price levels above VAH where only 1 TPO period traded (initiative buying strength)
  - Single prints below VA: same concept below VAL (initiative selling strength)
  - Poor high flag: if the highest TPO period has >2 letters (excess TPO), the high is "poor" and likely to be revisited
  - Poor low flag: same for the low
- [ ] Merge AMT features with Pomorski features:
  - Align on date index
  - Handle missing AMT data (if intraday data is unavailable for a day, forward-fill or mark as NaN)
  - Apply fractional differencing to non-stationary AMT features (VA width, VWAP level, delta levels)
  - Lag all AMT features by 1 day
- [ ] Save merged feature matrix: `data/features/spy_all_features.parquet`
- [ ] Verify: correlation analysis between Pomorski and AMT feature groups. Low correlation = good (new information). High correlation = redundancy to prune

**Definition of done:** Unified daily feature matrix with ~40-50 features (Pomorski + AMT), all lagged, fractionally differenced where needed, saved to disk. Correlation analysis confirms AMT features are not purely redundant.

**Knowledge required:** VWAP calculation, volume delta concept (approximate), AMT concepts of initiative/responsive activity, poor highs/lows, single prints.

**Tools required:** Same as Week 7. No new dependencies.

---

### Week 9 — AMT-Enhanced Classifier and Comparative Analysis

**What you're doing:** Retraining the classifier with AMT features included and rigorously measuring whether they improve performance.

**Tasks:**

- [ ] Retrain RF classifier on Pomorski + AMT features:
  - Same walk-forward setup as Stage 1
  - Same hyperparameters initially (retune if time permits)
  - 6-class regime labels
- [ ] Run BorutaSHAP on the expanded feature set:
  - Which AMT features are confirmed as important?
  - Which are rejected?
  - Prune rejected features
- [ ] Comparative evaluation (AMT vs no-AMT):
  - MCC per class: Stage 1 (Pomorski only) vs Stage 2 (Pomorski + AMT)
  - Target: MCC improvement ≥ 0.05 on at least 2 regime classes
  - Financial metrics: Sortino, cumulative returns, max drawdown — both setups on same validation period
  - Statistical test: are the MCC differences significant? Use paired t-test across walk-forward folds
- [ ] PCA dimensionality analysis:
  - PCA on Pomorski features only: how many components for 95% variance?
  - PCA on Pomorski + AMT features: how many components for 95% variance?
  - If the AMT-included model requires more components, the features add genuinely new information
  - Report the additional dimensions and which AMT features load most heavily on them
- [ ] SHAP analysis on AMT-enhanced model:
  - Are any AMT features in the top 15?
  - Which AMT features have the highest importance for each regime?
  - Expected: VA migration and delta divergence should be important for transition-related regimes
- [ ] Decision point:
  - If AMT features improve MCC and appear in SHAP top features → proceed with enhanced model
  - If AMT features don't help → diagnose (data quality? feature engineering? intraday data coverage?) or proceed with Pomorski-only model and revisit AMT later with better data
- [ ] Document comparative results in `notebooks/stage2_amt_comparison.ipynb`

**Definition of done:** Clear, documented answer to "Do AMT features improve regime classification?" with supporting metrics, SHAP plots, and PCA analysis. Decision made on whether to keep AMT features.

**Knowledge required:** PCA interpretation, comparative statistical testing.

**Tools required:** `scikit-learn` (PCA), `shap`, `scipy` (paired t-test).

---

## Stage 2 Gate

**Before proceeding to Stage 3, you must have:**

- [ ] Final feature set confirmed (Pomorski-only or Pomorski + AMT)
- [ ] 6-class classifier (or justified collapse to fewer classes) with MCC > 0.25 on actionable classes
- [ ] Understanding of what each feature contributes via SHAP
- [ ] Documented PCA comparison if AMT features were tested
- [ ] A robust, cached feature pipeline that runs in under 5 minutes for the full history

---

## Stage 3: Per-Regime Strategy Optimisation

**Goal:** Replace the naive long/short/flat strategy with optimised, parameterised strategy templates per regime. Measure whether per-regime strategies outperform the simple approach.

---

### Week 10 — Strategy Template Design and Implementation

**What you're doing:** Coding the parameterised strategy templates for each regime. Not optimising yet — just making them functional with sensible defaults.

**Tasks:**

- [ ] Define the strategy interface (all templates share the same API):
  ```python
  class RegimeStrategy:
      def generate_signal(self, features, regime_proba, current_position, amt_levels) -> Signal
      def compute_stop(self, entry_price, amt_levels, regime_confidence) -> float
      def compute_position_size(self, regime_confidence, regime_age, account_risk) -> float
  ```
- [ ] Implement **TrendFollowingLong** (for Trending Up regime):
  - Entry: buy when price pulls back to VWAP or developing VAL (if available), or when ROC > threshold after pullback
  - Exit: trailing stop at N × ATR below highest close since entry
  - Parameters: `pullback_threshold`, `trailing_stop_atr_mult` (default: 2.0), `min_holding_period` (default: 5 days)
- [ ] Implement **TrendFollowingShort / Defensive** (for Trending Down regime):
  - Entry: short on rally to VWAP or developing VAH, or go flat/cash
  - Exit: trailing stop at N × ATR above lowest close since entry
  - Parameters: `rally_threshold`, `trailing_stop_atr_mult`, `min_holding_period`
  - Alternative (simpler): just go to cash/flat when regime is Down. Implement short version as optional
- [ ] Implement **MeanReversion** (for Ranging Neutral regime):
  - Entry: long when price touches VAL (or lower Bollinger Band if no AMT data), short when price touches VAH (or upper BB)
  - Exit: take profit at POC or opposite VA boundary. Stop beyond VA + 1 ATR buffer
  - Parameters: `entry_buffer_atr` (how close to VA boundary to trigger), `take_profit_target` (POC or opposite VA), `stop_buffer_atr` (default: 1.0)
- [ ] Implement **DistributionRange** (for Ranging After Uptrend):
  - Same as MeanReversion but with tighter stops and position size that decays with regime age
  - Additional: flag if delta at VAH turns negative (initiative selling = potential breakdown)
  - Parameters: inherits from MeanReversion + `age_decay_rate` (default: reduce size 2% per day in regime)
- [ ] Implement **AccumulationRange** (for Ranging After Downtrend):
  - Same as MeanReversion but biased long (buy at VAL with wider stop, short at VAH with tighter stop)
  - Additional: flag if delta at VAL turns positive (initiative buying = potential breakout)
  - Parameters: inherits from MeanReversion + `long_bias_factor` (default: 1.5× size on longs vs shorts)
- [ ] Implement **TransitionWait** (for Transition/Breakout regime):
  - No new entries. Existing positions managed by stops only
  - If no position: wait until regime confidence > 0.6 in a non-transition state
  - Parameters: `confirmation_threshold` (default: 0.6), `max_wait_days` (default: 10, then go flat)
- [ ] Implement position sizing module:
  - Base size: fixed fractional (2% account risk per trade)
  - Confidence scaling: full size at regime probability > 0.6, half at 0.4-0.6, zero below 0.4
  - Regime age scaling: 100% at regime start, decaying by `age_decay_rate` per day
  - Combine: `final_size = base × confidence_scale × age_scale`
- [ ] Implement circuit breaker:
  - If current bar range > 2 × ATR(14) AND volume > 1.5 × 20-day average: override to Transition/Breakout regardless of classifier output
  - Close any mean-reversion positions immediately
- [ ] Unit test each strategy template with synthetic price data to verify entry/exit logic works correctly

**Definition of done:** All 6 strategy templates implemented with default parameters, unit tested, and producing correct signals on synthetic data. Position sizing and circuit breaker functional.

**Knowledge required:** Basic trading strategy implementation, stop loss mechanics, position sizing.

**Tools required:** Just Python. No new dependencies.

---

### Week 11 — Strategy Parameter Optimisation

**What you're doing:** Optimising the 3-5 parameters per strategy template using historical regime episodes.

**Tasks:**

- [ ] Extract historical regime episodes:
  - For each regime in the labelled history, identify start/end dates
  - Collect the price data, features, and AMT levels for each episode
  - Store as a list of episode DataFrames per regime
- [ ] Set up per-regime optimisation with Optuna:
  - For each regime-strategy pair:
    - Objective: maximise Sortino ratio on walk-forward validation episodes
    - Run the strategy template on training episodes with candidate parameters
    - Evaluate on validation episodes
    - 100 trials per regime (total: 600 trials across 6 regimes)
  - Constraint: each regime must have ≥ 30 episodes for meaningful optimisation. If fewer, use default parameters and do not optimise
- [ ] Optimise each regime-strategy pair:
  - TrendFollowingLong: `pullback_threshold`, `trailing_stop_atr_mult`
  - TrendFollowingShort: `rally_threshold`, `trailing_stop_atr_mult`
  - MeanReversion: `entry_buffer_atr`, `take_profit_target`, `stop_buffer_atr`
  - DistributionRange: inherits from MeanReversion + `age_decay_rate`
  - AccumulationRange: inherits from MeanReversion + `long_bias_factor`
  - TransitionWait: `confirmation_threshold`
- [ ] Evaluate optimised vs default parameters:
  - Per-regime Sortino ratio improvement
  - Overall system Sortino improvement
  - Check: did optimisation improve validation performance without destroying out-of-sample? Compare train vs validation Sortino ratios. If train >> validation, you've overfit the strategy parameters
- [ ] Save optimal parameters to `configs/strategy_params.json`

**Definition of done:** Each regime-strategy pair has optimised parameters validated on out-of-sample episodes. Parameters documented and saved.

**Knowledge required:** Optuna objective function setup, understanding of Sortino ratio as objective.

**Tools required:** `optuna`.

---

### Week 12 — Full System Backtest and Stage 3 Alpha Measurement

**What you're doing:** Running the complete system end-to-end (features → classifier → per-regime strategies) and measuring alpha rigorously.

**Tasks:**

- [ ] Build the full system backtest:
  - Daily loop:
    1. Compute features from available data (Pomorski + AMT if applicable)
    2. Run classifier → get regime probability vector
    3. Check circuit breaker conditions
    4. Select active strategy based on regime classification (or probability-weighted blend)
    5. Generate signal from active strategy
    6. Apply position sizing (confidence × regime age)
    7. Execute trade, update positions, deduct costs
    8. Log everything: regime prediction, confidence, strategy active, position, PnL
  - Walk-forward: retrain classifier every 6 months
- [ ] Run on full validation period (2018-2022 or similar):
  - Generate equity curve, trade log, regime prediction timeline
  - Compute all financial metrics: Sortino, Sharpe, cumulative returns, max drawdown, win rate, profit factor
- [ ] Compare Stage 3 vs Stage 1:
  - Per-regime strategies vs simple long/short/flat
  - Does the strategy layer add alpha beyond what the classifier provides?
  - Run Test C from attribution matrix: classifier + simple long/short
  - If Stage 3 Sortino > Stage 1 Sortino significantly, per-regime strategies are contributing
- [ ] Run Fama-French factor regression on Stage 3 returns:
  - α still positive and significant?
  - Has momentum loading (β₆) changed? If β₆ decreased while α stayed high, your strategies are capturing non-momentum alpha
- [ ] Regime-conditional factor regression:
  - Run separate regressions for periods classified as each regime
  - Which regimes produce the most alpha?
  - Expected: ranging regimes (where AMT-informed mean reversion is deployed) should show the most alpha
- [ ] Bootstrap the full system alpha (10,000 resamples)
- [ ] Document everything in `notebooks/stage3_full_system.ipynb`

**Definition of done:** Complete system evaluated with factor-regression-validated alpha. Clear attribution of which regimes and strategies contribute. Comparison against Stage 1 baseline documented.

**Knowledge required:** Same as Week 6, applied to the full system.

**Tools required:** Same as Week 6.

---

## Stage 3 Gate

**Before proceeding to Stage 4, you must have:**

- [ ] Sortino > 2.0 after costs on validation data
- [ ] Statistically significant alpha in Fama-French regression (p < 0.05)
- [ ] Per-regime strategies outperform naive long/short/flat (Stage 3 > Stage 1)
- [ ] At least 2 regime-strategy pairs independently profitable
- [ ] Max drawdown < 30%
- [ ] Strategy does significantly better than random regime labels (Test A >> Test D)

---

## Stage 4: Holdout Test and Robustness

**Goal:** Touch the test set for the first time. Measure out-of-sample performance and robustness. This is the moment of truth.

---

### Week 13 — Holdout Evaluation

**What you're doing:** Running the fully trained, frozen system on data it has never seen. No parameter changes allowed after this point.

**Tasks:**

- [ ] Freeze the system:
  - Final classifier model (trained on all data up to holdout start)
  - Final strategy parameters (from Week 11 optimisation)
  - Final feature set (from BorutaSHAP)
  - No changes to any of the above during holdout evaluation
- [ ] Run on holdout set (2022-2024 or whatever was reserved):
  - Full daily loop as in Week 12
  - Record all predictions, positions, PnL
- [ ] Compute holdout metrics:
  - Sortino, Sharpe, cumulative returns, max drawdown, win rate
  - Compare to validation metrics: how much degradation?
  - Target: holdout Sortino > 50% of validation Sortino (some decay is expected and healthy)
- [ ] Fama-French regression on holdout returns:
  - α still positive and significant?
  - If α is positive but not significant (small sample), note it — 2 years of daily data is ~500 observations, may lack power
- [ ] Regime-conditional analysis on holdout:
  - Did each regime-strategy pair behave as expected?
  - Any regime that was profitable in validation but not in holdout?
- [ ] Stress test specific periods:
  - 2022 rate shock: did the system correctly identify the trending-down regime and either short or go defensive?
  - Any flash crashes or gap days in the holdout: did the circuit breaker fire? Did stops protect capital?
- [ ] Document holdout results in `notebooks/stage4_holdout.ipynb`

**Definition of done:** Holdout results documented. Honest assessment of out-of-sample performance. Decision on whether to proceed to paper trading.

---

### Week 14 — Robustness Analysis and Sensitivity Testing

**What you're doing:** Stress-testing every assumption and parameter to understand where the system is fragile.

**Tasks:**

- [ ] Labeller sensitivity:
  - Rerun full pipeline with labeller MA lookback = 40, 50, 60
  - How much does holdout performance change? If Sortino swings from 3.0 to 0.5, the system is fragile
- [ ] Feature sensitivity:
  - Drop the top 3 SHAP features one at a time. Retrain and evaluate
  - If performance collapses when one feature is removed, the system is over-reliant on it
- [ ] Transaction cost sensitivity:
  - Rerun backtest with costs at 0.1%, 0.2%, 0.4%, 0.8% round-trip
  - At what cost level does alpha disappear? That's your break-even cost
- [ ] Regime duration sensitivity:
  - Rerun with minimum regime duration = 3, 5, 10, 15 days
  - How does turnover and Sortino change?
- [ ] Rolling alpha stability:
  - Compute rolling 6-month alpha (Fama-French intercept)
  - Plot over time. Is alpha stable or concentrated in specific periods?
  - Target: positive in > 60% of rolling windows
- [ ] Different instrument test (optional but valuable):
  - Run the same pipeline on QQQ or IWM using the same hyperparameters (no retuning)
  - If it works on a different instrument without retuning, the approach generalises
  - If it fails, the system may be overfit to SPY's specific characteristics
- [ ] Document all sensitivity results and identify the weakest assumptions

**Definition of done:** Comprehensive sensitivity analysis documented. Clear understanding of which parameters/assumptions the system is most sensitive to. Risk register updated.

---

## Stage 4 Gate

**Before proceeding to paper trading:**

- [ ] Holdout Sortino > 1.0 after costs
- [ ] Holdout alpha positive (even if not significant due to sample size)
- [ ] No catastrophic sensitivity to any single parameter or feature
- [ ] Rolling alpha positive in > 60% of windows
- [ ] Max drawdown in holdout < 30%
- [ ] Honest assessment written: "Would I risk real money on this?"

---

## Stage 5: Paper Trading

**Goal:** Validate forward performance on live data without risking capital.

---

### Week 15 — Broker Integration and Automation

**What you're doing:** Connecting the system to a broker API and automating daily execution.

**Tasks:**

- [ ] Choose broker:
  - Alpaca: simplest API, good for equities, free paper trading
  - IBKR: more instruments (futures, FX), harder API, also has paper trading mode
  - Recommendation: start with Alpaca for SPY paper trading
- [ ] Implement daily execution script:
  - Runs at 16:30 ET (after market close)
  - Downloads today's OHLCV data
  - Computes features (including AMT features if using intraday data)
  - Runs classifier → regime prediction
  - Checks circuit breaker conditions
  - Generates signal from active strategy
  - Computes position sizing
  - Places orders for next session's open via broker API
  - Logs everything to a daily log file
- [ ] Implement stop loss order management:
  - Place stops as broker-side orders (not managed by the script)
  - Update stops daily based on trailing stop logic and regime confidence
  - Critical: stops must persist even if the script crashes
- [ ] Set up as a cron job or scheduled task:
  - Runs automatically each trading day
  - Sends email/Slack notification with today's regime prediction, confidence, and actions taken
  - Sends alert if any error occurs
- [ ] Test thoroughly in paper trading mode:
  - Verify orders are placed correctly
  - Verify stops are set correctly
  - Verify position sizing matches expectations
  - Run for 1 week manually monitoring every action before going fully automated

**Definition of done:** Automated daily pipeline running in paper trading mode. Orders placing correctly. Stops managed at broker level. Notifications working.

**Knowledge required:** Broker API (Alpaca or IBKR), cron jobs, basic error handling/alerting.

**Tools required:** `alpaca-trade-api` or `ib_insync`, cron/launchd, email/Slack webhook.

---

### Weeks 16-28 — Paper Trading Period (3 months minimum)

**What you're doing:** Letting the system run and collecting forward performance data. Minimal intervention.

**Tasks:**

- [ ] Daily monitoring (5 minutes/day):
  - Review today's regime prediction and confidence
  - Check that orders executed correctly
  - Note any anomalies in the log
  - Do NOT change parameters based on short-term results
- [ ] Weekly review (30 minutes/week):
  - Compute rolling Sortino and cumulative returns
  - Compare to buy-and-hold over the same period
  - Check regime prediction accuracy: does the classified regime match what the market is actually doing?
  - Log any regime transitions the system detected late or missed entirely
- [ ] Monthly review (2 hours/month):
  - Full performance report: Sortino, Sharpe, cumulative returns, max drawdown, trade count
  - Factor regression on the paper trading period returns
  - Compare to backtested performance: is forward alpha within 50% of backtest alpha?
  - Regime-by-regime analysis: which strategies are contributing? Which are not?
  - Decision point: if forward Sortino is negative after 2 months, pause and diagnose
- [ ] At end of 3-month paper trading:
  - Full evaluation report comparing paper trading results to holdout backtest results
  - Factor regression on forward returns
  - Bootstrap forward alpha
  - Honest final assessment: "Is this system ready for real capital?"

**Definition of done:** 3+ months of paper trading data collected. Forward performance documented and compared to backtest expectations.

---

## Final Gate — Live Decision

**Criteria for proceeding to live trading with real capital:**

- [ ] Forward Sortino > 0.5 (lower bar than backtest due to small sample)
- [ ] Forward returns positive after costs
- [ ] No position-level loss exceeding 3× expected from backtest
- [ ] Max drawdown in paper trading < max drawdown in holdout backtest
- [ ] System ran reliably (no missed days, no order failures, no data gaps)
- [ ] Factor regression alpha positive (may not be significant with only 3 months)
- [ ] Your honest assessment: the system is behaving as expected and you understand its risks

**If you pass:** begin live trading with minimal capital (10-20% of intended allocation). Scale up over 3-6 months as forward performance confirms.

**If you don't pass:** identify what diverged from expectations. Is it the classifier? The strategies? The market regime? Iterate on the weakest component and re-enter paper trading.

---

## Timeline Summary

| Week | Stage | Focus | Key Deliverable |
|------|-------|-------|-----------------|
| 1 | Stage 1 | Environment + data | Clean daily data pipeline |
| 2 | Stage 1 | Feature engineering | ~25 Pomorski-style features |
| 3 | Stage 1 | Regime labeller | 6-state labels, visually validated |
| 4 | Stage 1 | Baseline classifier | 3-class RF, MCC > 0.3 |
| 5 | Stage 1 | Full classifier + tuning | 6-class RF, hyperparameters tuned |
| 6 | Stage 1 | **Stage 1 backtest + alpha test** | **Gate: does baseline system have alpha?** |
| 7 | Stage 2 | Intraday data + TPO/VP | AMT feature engine (part 1) |
| 8 | Stage 2 | VWAP, delta, OI features | AMT feature engine (part 2) |
| 9 | Stage 2 | **AMT classifier comparison** | **Gate: do AMT features help?** |
| 10 | Stage 3 | Strategy templates | 6 parameterised strategies |
| 11 | Stage 3 | Strategy optimisation | Per-regime parameters tuned |
| 12 | Stage 3 | **Full system backtest** | **Gate: does full system produce alpha?** |
| 13 | Stage 4 | Holdout evaluation | First touch of test data |
| 14 | Stage 4 | **Robustness testing** | **Gate: is the system robust?** |
| 15 | Stage 5 | Broker integration | Automated paper trading pipeline |
| 16-28 | Stage 5 | **Paper trading (3 months)** | **Final gate: forward performance** |
