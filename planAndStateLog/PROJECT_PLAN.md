# AMT-Regime Quantitative Algorithm — Project Plan

## 1. Project Overview

This project builds a quantitative trading system that decomposes financial markets into discrete structural regimes using Auction Market Theory (AMT) and deploys optimised strategies per regime. The core thesis is that markets cycle through identifiable behavioural states — trending, ranging, and transitional — and that strategies tuned to each state will outperform static approaches.

The architecture extends the work of Pomorski (2024), *"Construction of Effective Regime-Switching Portfolios Using a Combination of Machine Learning and Traditional Approaches"* (UCL PhD thesis), which validated a detection-prediction-optimisation pipeline using KAMA+MSR for regime labelling, Random Forest for regime prediction, and Model Predictive Control for portfolio construction. This project diverges from Pomorski by:

- Expanding from 4 regimes (volatility × trend) to 6 regimes incorporating directional context and AMT structural states.
- Grounding regime definitions in Auction Market Theory rather than pure statistical volatility decomposition.
- Integrating market microstructure data (TPO, volume profile, delta, VWAP, open interest) as classifier features alongside traditional momentum and volatility indicators.
- Deploying distinct, optimised strategy templates per regime rather than a single long/short approach.
- Using AMT structural levels (value area high/low, POC) for execution-level risk management, including adaptive stop losses that act as real-time regime change detectors.

The predicted edge is structural alpha from two sources: (1) improved regime transition detection via AMT microstructure features that capture participant behaviour, not just price action, and (2) per-regime strategy optimisation that exploits the distinct statistical properties of each market state.

---

## 2. Goals

### Primary Goals

1. **Build a regime classifier** that identifies 6 market states with MCC > 0.3 per class on out-of-sample data using walk-forward validation.
2. **Develop optimised strategy templates** for each regime that are independently profitable after transaction costs.
3. **Demonstrate statistically significant alpha** after controlling for market beta and momentum factors (Fama-French 5-factor + momentum regression, p < 0.05 on intercept).
4. **Maintain robustness** across multiple market cycles (2008 GFC, 2020 COVID, 2022 rate shock) and across asset classes (equities, commodities, FX).

### Secondary Goals

5. Validate that AMT microstructure features add information beyond price-derived features (measurable via PCA dimensionality increase and SHAP importance).
6. Achieve regime transition detection within 1-3 bars of actual transition via probability monitoring and circuit breakers.
7. Paper trade the system for 3-6 months with forward alpha consistent with backtested alpha (within 50% of backtest estimate).

### Non-Goals

- Sub-second or HFT latency. This is a structural alpha strategy operating at daily frequency.
- Replacing human judgement on macro regime shifts. The system is a tool, not fully autonomous.
- Capturing every regime transition perfectly. The goal is making latency non-fatal, not zero.

---

## 3. The Six Regime States

The regime model is grounded in Auction Market Theory, which describes markets as two-way auctions that alternate between balance (fair value discovery) and imbalance (directional price discovery). The Wyckoff price cycle (accumulation → markup → distribution → markdown) provides the structural foundation, extended with contextual transitions.

| # | Regime | AMT Interpretation | Characteristics | Duration |
|---|--------|--------------------|-----------------|----------|
| 1 | **Trending Up** | Imbalance — initiative buying driving price discovery higher | Rising value area, responsive buying at pullbacks, single prints below, expanding range | Days to months |
| 2 | **Trending Down** | Imbalance — initiative selling driving price discovery lower | Falling value area, responsive selling at rallies, single prints above, expanding range | Days to months |
| 3 | **Ranging (Neutral)** | Balance — two-way trade within established value area | Symmetric TPO distribution, stable POC, volume concentrated at centre, low directional delta | Days to weeks |
| 4 | **Ranging After Uptrend (Distribution)** | Balance after imbalance — potential distribution phase | Value area stops migrating up, volume shifts to upper range, initiative selling appears at highs, possible poor highs forming | Days to weeks |
| 5 | **Ranging After Downtrend (Accumulation)** | Balance after imbalance — potential accumulation phase | Value area stops migrating down, volume shifts to lower range, initiative buying appears at lows, possible poor lows forming | Days to weeks |
| 6 | **Transition / Breakout** | Imbalance initiation — auction resolving directionally from balance | Rapid value area expansion, volume surge, delta divergence resolving, single prints forming in direction of breakout | Hours to days |

### State Transition Logic

```
Trending Up ──→ Ranging After Uptrend (Distribution) ──→ Trending Down (breakdown)
                                                       ──→ Trending Up (continuation)
                                                       ──→ Ranging Neutral (extended balance)

Trending Down ──→ Ranging After Downtrend (Accumulation) ──→ Trending Up (breakout)
                                                          ──→ Trending Down (continuation)
                                                          ──→ Ranging Neutral (extended balance)

Ranging Neutral ──→ Transition/Breakout ──→ Trending Up or Trending Down

Transition/Breakout ──→ Trending Up or Trending Down (confirmed)
                     ──→ Ranging (failed breakout / false break)
```

### Minimum Regime Duration Constraint

Regimes lasting fewer than 5 trading days are filtered as noise. This prevents excessive strategy switching and the transaction cost drag that comes with it. The Transition/Breakout state is the exception — it may last 1-3 days by nature and serves as a buffer between sustained states.

---

## 4. Algorithm Architecture

The system is a four-layer pipeline. Each layer operates independently and communicates via well-defined interfaces.

### Layer 1: Feature Engineering

**Input:** Raw OHLCV data (daily + intraday), macro data, AMT microstructure data.

**Output:** A daily feature vector per instrument (~40-50 features after selection).

**Pomorski-style features (price-derived):**

| Category | Features | Source |
|----------|----------|--------|
| Momentum | ROC at 1M, 3M, 6M, 12M lookbacks | Daily close |
| Trend | SMA crossovers (50/200), KAMA, ADX, Plus/Minus DI | Daily close |
| Volatility | ATR (14d, 30d), rolling std (20d, 60d), VIX level, VIX change | Daily close, CBOE |
| Mean-reversion | RSI (14d), Bollinger Band %B, CCI, CMO | Daily close |
| Volume | OBV, volume ratio (current/20d avg), MFI, Force Index | Daily OHLCV |
| Stationarity | ADF test statistic (rolling 252d), ADF p-value | Daily close |
| Time-series | Time reversal asymmetry (lags 1-3), energy ratio by chunks | Daily close |
| Macro | 10Y yield, yield curve slope (10Y-2Y), VIX term structure | FRED, CBOE |

**AMT microstructure features (session-derived):**

| Category | Features | Source |
|----------|----------|--------|
| TPO | Value area width (% of price), VA migration (1d, 5d), TPO count at POC, single prints above/below VA, poor high/low flags, distribution shape (b/P/D/p) | 30-min intraday bars |
| Volume Profile | POC vs close, POC migration rate, HVN count in range, nearest LVN above/below, volume concentration ratio | Intraday bars (5-30 min) |
| VWAP | Price vs session VWAP, price vs weekly VWAP, VWAP slope (5d), VWAP touch count (5d) | Intraday bars |
| Volume Delta | Cumulative delta (5d), delta divergence from price, delta at VAH, delta at VAL, delta rate of change | Intraday bars (approximated from bar direction) or trade-level data |
| Open Interest | OI change (1d), OI vs price divergence, OI percentile (30d) | CME daily settlement (futures only) |

**Preprocessing:**

- Fractional differencing: find minimum d per feature that passes KPSS/ADF at 5% significance. Preserves memory while achieving stationarity. Typical d ≈ 0.3-0.5.
- Normalisation: all AMT features expressed as ratios or ATR multiples (VA width as % of price, delta as % of total volume, POC distance as ATR multiple). Ensures cross-instrument and cross-time comparability.
- Feature selection: BorutaSHAP algorithm to prune redundant features. Expected to reduce ~50 raw features to ~25-30 informative features.
- Lookahead prevention: every feature lagged by minimum 1 day. All intraday-derived features computed from prior session only.

### Layer 2: Regime Classification

**Input:** Daily feature vector.

**Output:** Regime probability vector (6 classes) per instrument.

**Model: Random Forest classifier.**

Rationale (validated by Pomorski):
- Handles non-linear feature interactions without explicit specification.
- Outputs class probabilities, not just labels — critical for soft transitions and confidence-based sizing.
- Resistant to overfitting relative to deep learning given available data volumes (~2,500-5,000 daily observations per instrument over 10-20 years).
- Interpretable via SHAP values — can verify AMT features are contributing.
- Fast to train (seconds on CPU, no GPU required).
- No stationarity assumptions on the feature-target relationship.

**Hyperparameters (Pomorski's optimals as starting point):**

| Parameter | Range | Pomorski Optimal |
|-----------|-------|------------------|
| n_estimators | 100-300 | 220-280 |
| max_depth | 3-15 | 3-13 |
| min_samples_split | 10-100 | 18-76 |
| min_samples_leaf | 30-100 | 60-95 |
| max_samples | 0.1-0.5 | 0.12-0.36 |
| max_features | 0.2-0.5 | 0.25-0.40 |

Optimisation via Optuna maximising Sortino ratio over walk-forward validation splits (PGTS — Purged Group Time Series cross-validation).

**Fallback model:** XGBoost/LightGBM if RF underperforms. Do not use deep learning unless data volume increases substantially (e.g., intraday regime classification with years of 5-min bars).

**Labelling (ground truth generation):**

Phase 1 — Rule-based labeller (v1):
1. Trend direction: price above/below adaptive MA (KAMA, parameters optimised per instrument).
2. Volatility regime: rolling ATR(14) above/below its 50-day SMA.
3. Prior state memory: what was the classified state 10 bars ago.
4. Combining (1), (2), and (3) produces 6 labels mechanically.

Phase 2 — KAMA+MSR labeller (v2):
- Implement Pomorski's KAMA+MSR model for 4-state detection.
- Extend to 6 states by splitting transitory states using prior-state context.
- More rigorous but more complex to implement. Use after v1 pipeline is working end-to-end.

### Layer 3: Strategy Execution

**Input:** Regime probability vector, current positions, AMT structural levels.

**Output:** Trade signals with position sizing.

**Strategy templates per regime:**

| Regime | Strategy Class | Entry Logic | Exit Logic | Key Parameters |
|--------|---------------|-------------|------------|----------------|
| Trending Up | Trend-following | Buy pullbacks to developing VAL or VWAP; breakout entries above prior session VAH | Trailing stop at developing VAL or N×ATR below high | Pullback depth threshold, trailing stop multiplier, position scale-in rules |
| Trending Down | Trend-following (short) / Defensive | Short rallies to developing VAH or VWAP; or rotate to cash/bonds | Trailing stop at developing VAH or N×ATR above low | Rally depth threshold, trailing stop multiplier, hedge ratio |
| Ranging Neutral | Mean-reversion | Fade at VA extremes (long VAL, short VAH) | Take profit at POC or opposite VA extreme; stop beyond VA + buffer | VA buffer size, take-profit ratio, max holding period |
| Ranging After Uptrend | Mean-reversion + breakout watch | Fade at VAH with tight stops; reduce size as regime ages | Stop above VAH + 1 ATR; if breakout confirmed, reverse to trend strategy | Stop tightness (tied to regime confidence), position size decay rate |
| Ranging After Downtrend | Mean-reversion + breakout watch | Buy at VAL with tight stops; reduce size as regime ages | Stop below VAL - 1 ATR; if breakout confirmed, reverse to trend strategy | Stop tightness (tied to regime confidence), position size decay rate |
| Transition / Breakout | Reduced exposure / wait for confirmation | No new positions until regime classifier confirms new state (>0.6 probability) | Existing positions managed by stops only | Confirmation threshold, maximum time in transition before defaulting to flat |

**Position sizing:**

- Base size determined by Kelly criterion or fixed fractional (2% risk per trade).
- Scaled by regime classifier confidence: full size at >0.6 probability, half size at 0.4-0.6, no new positions below 0.4.
- Scaled inversely by regime age: fresh regime = 100% of base size, 30+ day regime = 70%, 60+ day regime = 50%. Accounts for increasing transition probability.
- Maximum portfolio heat: 6% total risk across all open positions.

**Adaptive stop losses (execution-level regime defence):**

Stops are set at AMT structural levels, not arbitrary distances:
- Ranging regimes: stop just beyond value area boundary + 1 ATR buffer.
- Trending regimes: trailing stop at developing value area boundary in direction of trend.
- Stop tightness modulated by classifier confidence: lower confidence = tighter stops.

When a stop is hit, it functions as a real-time regime change signal at the execution level, independent of the classifier. The position is closed immediately. The classifier catches up within 1-3 bars and deploys the appropriate new strategy.

### Layer 4: Portfolio Construction and Risk Management

**Input:** Trade signals from Layer 3 across all instruments.

**Output:** Final portfolio weights and orders.

For a single-instrument version (Phase 1), this layer is minimal — just position sizing and stop management. For multi-instrument/multi-asset (Phase 2+), implement:

- Correlation-aware position sizing: reduce aggregate exposure when instrument correlations spike (regime changes tend to increase cross-asset correlation).
- Regime-conditional allocation: overweight instruments with high-confidence regime classifications, underweight uncertain ones.
- Optional: Model Predictive Control (Pomorski Chapter 5) for multi-period optimisation with Kalman filter return estimation. Implement only after single-instrument system is validated.

---

## 5. Regime Transition Handling

Regime transitions are where the most money is made or lost. The system uses a six-layer defence against transition latency.

### 5.1 Predictive Classification (not reactive)

The RF classifier is trained to predict tomorrow's regime from today's features. Pre-breakout signatures (volatility compression, declining volume, range narrowing, time-reversal asymmetry shifts) are captured in the feature set. The classifier fires before the breakout, not after.

### 5.2 Soft Probability Thresholds

The classifier outputs a 6-class probability vector. Strategy deployment is graduated:

| Confidence Level | Action |
|-----------------|--------|
| Dominant regime probability > 0.6 | Full conviction — deploy that regime's strategy at full size |
| Dominant regime probability 0.4-0.6 | Blended response — begin scaling into new strategy, scale out of old |
| No regime > 0.4 | Low conviction — reduce position size, widen stops, or sit out |

### 5.3 Regime Probability Rate of Change

Track the velocity of probability shifts:

```
regime_momentum = p_regime_today - p_regime_yesterday
```

Fast-declining probability for the current regime is an early warning even before the formal classification flips. Fed back into position sizing (faster decline = smaller positions in current regime strategy).

### 5.4 Multi-Timeframe Disagreement

Run feature computation at multiple timeframes (daily + 4H + 1H). When shorter timeframes diverge from longer (e.g., daily says ranging, hourly momentum features say breakout), flag as elevated transition risk. Encode as a feature ("timeframe disagreement score") or use directly for position sizing.

### 5.5 Circuit Breaker Override

Hard rule that overrides the classifier for extreme moves:

- If classified regime is "ranging" AND price moves > 2× ATR in a single session AND volume > 1.5× 20-day average: force-flag as potential regime transition.
- Immediately suspend mean-reversion strategy, go flat or to minimum size.
- Wait for classifier confirmation within 1-3 bars before deploying new strategy.

Cost: occasional false positives on spike-and-reverse days. Benefit: prevents catastrophic loss from fading a genuine breakout.

### 5.6 Strategy-Level Safe Failure

Each strategy is designed to fail safely when the regime changes unexpectedly:

- Mean-reversion strategies always have stops at structural levels. Stop hit = small loss, not catastrophe.
- Trend strategies use trailing stops. Trend ending = profit lock, not loss.
- Position size decays with regime age, so exposure is naturally smaller when transitions are most likely.

---

## 6. Data Requirements

### 6.1 Data Sources

| Data Type | Source | Cost | Priority |
|-----------|--------|------|----------|
| Daily OHLCV (equities) | yfinance, Alpaca, Polygon.io free tier | Free | P0 — required |
| Daily OHLCV (futures/commodities) | Databento, CME DataMine | Free-$100/mo | P0 — required |
| VIX (daily) | CBOE (via yfinance) | Free | P0 — required |
| US Treasury yields (10Y, 2Y, 3M) | FRED API | Free | P0 — required |
| Intraday bars 5-30 min (equities) | Polygon.io, Alpaca | Free-$30/mo | P1 — needed for AMT features |
| Intraday bars (futures) | Databento | $50-100/mo | P1 — needed for AMT features |
| Open interest (daily, futures) | CME daily settlement | Free (delayed 1 day) | P2 — nice to have |
| Trade-level data (for true delta) | Databento, Rithmic | $100-200/mo | P3 — upgrade from bar-level delta |
| Level 2 / orderflow | Rithmic, CQG, TT | $500+/mo | P4 — future enhancement |

### 6.2 Data Specifications

- **History required:** minimum 10 years daily (2014-2024) for training. 15-20 years preferred to capture 2008 GFC.
- **Intraday history:** minimum 3-5 years for AMT feature computation. Longer if available.
- **Update frequency:** daily at market close. No real-time streaming required for v1.
- **Storage estimate:** daily OHLCV for 25 instruments × 20 years ≈ 5MB. Intraday 5-min bars for 25 instruments × 5 years ≈ 2-5GB. Manageable on local disk.
- **Session definition:** RTH (Regular Trading Hours) only for TPO and volume profile computation. This is standard AMT practice and avoids Globex overnight noise.

### 6.3 Data Quality Constraints

- Survivorship bias: only use instruments that existed throughout the full training period. Do not include delisted stocks.
- Dividend/split adjustment: use adjusted close prices for all equity OHLCV.
- Futures roll: use continuous front-month contracts with back-adjustment for price continuity. Volume and OI from front month only.
- Missing data: forward-fill gaps of 1-2 days (holidays, halts). Drop instruments with >5% missing data.

---

## 7. Compute, Memory, and Infrastructure Constraints

### 7.1 Compute Requirements

| Component | Compute | Estimated Time |
|-----------|---------|----------------|
| Feature engineering (daily, 25 instruments) | CPU, single core | < 30 seconds |
| Feature engineering (intraday AMT, 25 instruments) | CPU, single core | 2-5 minutes |
| RF training (250 trees, 30 features, 5000 samples) | CPU, single core | < 10 seconds |
| XGBoost training (same scale) | CPU, single core | < 5 seconds |
| Walk-forward backtest (10 years, 25 instruments) | CPU, single core | 5-30 minutes |
| Hyperparameter optimisation (Optuna, 200 trials) | CPU, multi-core | 30-60 minutes |
| Full pipeline (feature eng + classify + backtest) | CPU, single core | < 10 minutes daily |

**No GPU required.** A modern laptop (M-series Mac or equivalent) is sufficient for all development and live execution.

### 7.2 Memory Requirements

| Component | RAM |
|-----------|-----|
| Daily feature matrix (25 instruments × 5000 days × 50 features) | ~50 MB |
| Intraday data in memory (1 instrument × 1 year × 5-min bars) | ~15 MB |
| Full intraday dataset (25 instruments × 5 years) | 2-5 GB on disk, loaded per-instrument |
| RF model in memory | < 100 MB |
| Backtest state (positions, equity curve, logs) | < 50 MB |

**Constraint:** do not load all intraday data into memory simultaneously. Pre-compute AMT features per instrument and cache as daily CSVs. Load only the daily feature matrix for classification and backtesting.

### 7.3 Infrastructure

**Development:** local machine, Python 3.10+, standard data science stack.

**Live execution (v1):** Python cron job running at market close. Computes features, runs classifier, generates signals, places orders via broker API (IBKR TWS API or Alpaca). No cloud infrastructure required.

**Live execution (v2):** if scaling to many instruments or intraday frequency, consider a small cloud instance (AWS t3.medium or equivalent, ~$30/month) running scheduled jobs.

### 7.4 Latency Requirements

| Component | Required Latency | Rationale |
|-----------|-----------------|-----------|
| Regime classification | End of day (minutes) | Structural alpha, not speed-based |
| Strategy signal generation | End of day (minutes) | Orders placed at next open or scheduled close |
| Order execution | Seconds to minutes | No HFT requirement; market/limit orders at open |
| Stop loss execution | Real-time during market hours | Broker-side stop orders, not algorithm-managed |
| Circuit breaker check | Intraday (hourly or at session checkpoints) | Catch extreme moves before classifier updates |

**Critical decision:** stops must be placed as broker-side orders, not managed by the algorithm. If the algorithm process dies, stops must still execute. This is a non-negotiable reliability requirement.

---

## 8. Technology Stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| Language | Python 3.10+ | Development speed, ecosystem, Pomorski used Python |
| Data handling | pandas, numpy | Standard, sufficient performance at this scale |
| Feature engineering | ta-lib, tsfresh, custom AMT module | ta-lib for standard indicators, custom code for AMT features |
| Fractional differencing | fracdiff or custom implementation | Following López de Prado (2018) |
| ML classifier | scikit-learn (RF), xgboost/lightgbm (fallback) | Proven, fast, interpretable |
| Feature selection | BorutaSHAP | Pomorski's approach, robust for RF |
| Hyperparameter tuning | Optuna | Bayesian optimisation, handles complex search spaces |
| Backtesting | vectorbt or custom walk-forward engine | vectorbt for speed; custom if AMT logic too complex |
| Explainability | SHAP | Feature importance, per-regime analysis |
| Factor analysis | statsmodels (OLS regression) | Fama-French factor regression for alpha measurement |
| Visualisation | matplotlib, plotly | Equity curves, regime overlays, feature importance |
| Broker API | IBKR TWS API or Alpaca | Live order execution |
| Data storage | CSV/Parquet files, SQLite for metadata | Simple, no database server needed |
| Model persistence | joblib | Save/load trained RF models between sessions |
| Version control | Git + GitHub | Already initialised |

---

## 9. Measures of Success and Definition of Alpha

### 9.1 Classification Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| Matthews Correlation Coefficient (MCC) per regime | > 0.3 (good), > 0.4 (strong) | Primary classifier accuracy metric, handles class imbalance |
| MCC average across "actionable" regimes (excl. Transition) | > 0.35 | Overall classifier quality |
| Regime transition detection rate | > 70% of transitions detected within 3 bars | Measures latency directly |
| Confusion matrix analysis | Trending-up never confused with Trending-down | Critical misclassifications identified |

### 9.2 Financial Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| Annualised Sortino ratio (after costs) | > 2.0 (good), > 4.0 (strong) | Primary risk-adjusted return metric; Pomorski achieved 4.7-25.0 |
| Annualised Adjusted Sharpe ratio | > 1.0 | Industry-standard risk-adjusted metric |
| Cumulative returns vs buy-and-hold | Must exceed passive holding after costs | Basic viability test |
| Information ratio vs benchmark | > 0.5 (decent), > 1.0 (strong) | Excess return per unit of tracking error |
| Maximum drawdown | < 20% (target), < 30% (acceptable) | Psychological and capital constraint |
| Win rate per regime-strategy pair | > 50% for each pair independently | Validates each regime strategy contributes |
| Profit factor | > 1.5 | Gross profit / gross loss |

### 9.3 Alpha Isolation

**Definition of alpha for this strategy:** the intercept (α) in a factor regression of strategy returns against known systematic risk factors, after transaction costs.

**Primary test — Fama-French 6-factor regression:**

```
R_strategy - R_f = α + β₁(MKT-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA) + β₆(MOM) + ε
```

- α must be positive and statistically significant (t-stat > 2.0, p < 0.05).
- β₆ (momentum) is the critical control. If α disappears when momentum is included, the regime detector is just a complex momentum signal, not a genuine edge.
- Factor data sourced from Kenneth French's data library (free).

**Regime-conditional factor regression:**

Run the above regression separately for each regime period. Identifies which regimes produce alpha and which are just beta exposure.

**Attribution test matrix:**

| Test | Classifier | Strategy | Purpose |
|------|-----------|----------|---------|
| A (full system) | Trained RF | Optimised per-regime | Benchmark: does the full system work? |
| B (oracle) | Perfect labels (hindsight) | Optimised per-regime | Upper bound: how good could it be? |
| C (naive strategy) | Trained RF | Simple long/short | Isolates classifier contribution |
| D (random labels) | Random assignment | Optimised per-regime | Lower bound / null hypothesis |

- A >> D: regime detection is informative.
- A ≈ B: classifier is good enough, strategy design is the bottleneck.
- A ≈ C: strategies aren't adding much, classifier is doing the work.
- A >> C and A >> D: both classifier and strategies contribute.

**AMT feature validation via PCA:**

```
PCA on Pomorski features only → N components for 95% variance
PCA on Pomorski + AMT features → M components for 95% variance
If M > N: AMT features add genuinely new information dimensions.
If M ≈ N: AMT features are redundant with price-derived features.
```

### 9.4 Statistical Robustness

| Test | Method | Threshold |
|------|--------|-----------|
| Bootstrap confidence | Resample returns 10,000×, compute alpha distribution | 5th percentile of alpha > 0 |
| Out-of-sample decay | Compare alpha in train / validation / test periods | Test alpha > 50% of validation alpha |
| Cross-cycle stability | Compute rolling 1-year alpha | Positive in >70% of rolling windows |
| Multiple testing correction | Bonferroni or Benjamini-Hochberg across all regime-strategy combinations | Adjusted p-values still < 0.05 |
| Paper trading validation | 3-6 months live paper trading | Forward alpha within 50% of backtested alpha |

---

## 10. Risks and Potential Issues

### 10.1 Overfitting

**The primary risk.** Multiple layers of optimisation (labeller parameters, feature selection, classifier hyperparameters, strategy parameters per regime) create compounding overfitting risk.

**Mitigations:**

- Walk-forward validation throughout. Never use random splits on time series.
- Strict train/validate/test separation. Test set untouched until final evaluation.
- PGTS (Purged Group Time Series) cross-validation to prevent information leakage between folds.
- Conservative model complexity: RF max_depth bounded, min_samples_leaf kept high (≥30).
- Monitor out-of-sample degradation at every stage. If validation-to-test decay exceeds 50%, the system is overfit.
- Fewer parameters per regime strategy (3-5 max). More parameters = more overfitting surface.
- Bonferroni correction for multiple testing across 6 regime-strategy combinations.
- Regime sample size constraint: if a regime has fewer than 50 occurrences in training data, collapse it into a broader category rather than fitting a sparse classifier.

### 10.2 Lookahead Bias

**Subtle and fatal.** Can creep in through:

- Features computed from data not yet available at prediction time (e.g., using today's close to predict today's regime).
- Labeller using future information to assign past regime labels (e.g., "this was accumulation because it subsequently broke out").
- Hyperparameters tuned on test data.

**Mitigations:**

- All features lagged by minimum 1 day. AMT features computed from prior session only.
- Rule-based labeller uses only backward-looking information (MA position, ATR level, prior state). The label at time t depends only on data up to time t.
- Hyperparameter tuning only on training/validation splits, never on test.
- Code review specifically checking for future data access in feature engineering pipeline.

### 10.3 Regime Label Quality

The classifier is only as good as its training labels. If the rule-based labeller produces noisy or inconsistent labels, the RF learns noise.

**Mitigations:**

- Visual inspection of labelled regimes overlaid on price charts for multiple instruments.
- Labeller stability test: small changes to labeller parameters (MA lookback ±5 days) should not dramatically change label distributions.
- Compare v1 (rule-based) and v2 (KAMA+MSR) labels. High agreement = robust definitions. Low agreement = label quality issue to resolve.
- Pomorski's K-Means misclassification score as a labeller quality metric.

### 10.4 Class Imbalance

Some regimes may be rare (Transition/Breakout may only appear 5-10% of the time). Standard classifiers will underpredict rare classes.

**Mitigations:**

- Use MCC instead of accuracy as the primary metric (handles imbalance).
- Class-weighted RF: set class_weight="balanced" or use custom weights inversely proportional to class frequency.
- If a regime has < 50 training samples, collapse it into the nearest broader category.
- SMOTE or oversampling as a last resort — but prefer natural data over synthetic.

### 10.5 Regime Definition Subjectivity

"Ranging after uptrend" vs "ranging neutral" involves human judgement about when a prior trend "ended." Different analysts would label the same period differently.

**Mitigations:**

- Mechanical labelling rules remove subjectivity from the labelling process.
- Sensitivity analysis: if relabelling ambiguous periods changes results significantly, the distinction is not robust enough to trade on. Collapse the ambiguous states.
- The classifier's own confusion matrix reveals which regime pairs are genuinely distinguishable and which are not.

### 10.6 Transaction Costs and Slippage

Backtested alpha evaporates when real-world costs are applied. Pomorski used 0.8% round-trip for equities, which is conservative.

**Mitigations:**

- Include transaction costs in every backtest from day one. Never evaluate cost-free results.
- Use Pomorski's cost model as a floor: 0.8% for equities, 0.27% for commodities, 0.13% for FX.
- For v2, implement Boyd et al.'s dynamic transaction cost function (Equation 51 in Pomorski) which accounts for volatility and volume.
- Minimum regime duration filter (5 days) prevents excessive switching.
- Monitor turnover. If annualised turnover exceeds 20×, costs are likely eroding alpha.

### 10.7 Market Regime Instability

Market microstructure changes over time (decimalization, algorithmic trading growth, passive flow dominance). A classifier trained on 2010-2020 data may not work in 2025.

**Mitigations:**

- Walk-forward retraining: retrain classifier periodically (quarterly or when performance degrades).
- Monitor classifier confidence distribution. If average confidence declines, the feature-regime relationship may be shifting.
- Use features robust to microstructure changes (momentum, volatility ratios) alongside AMT features.
- Paper trading validation before committing real capital.

### 10.8 Data Quality and Availability

AMT features require intraday data which is harder to source, more expensive, and more prone to quality issues (missing bars, incorrect volumes, exchange outages).

**Mitigations:**

- Start with Pomorski features only (daily data, free). Add AMT features incrementally.
- Validate intraday data quality before use: check for gaps, zero-volume bars, price spikes.
- Maintain fallback: if AMT data is unavailable for a session, use Pomorski-only features for that day.
- Pre-compute and cache AMT features daily. Do not depend on real-time intraday data access for the classifier.

---

## 11. Flexibility and Adaptability

### 11.1 Modular Architecture

Each layer (feature engineering, classification, strategy, portfolio) is independent with clean interfaces. This enables:

- Swapping the classifier (RF → XGBoost → neural network) without changing the feature pipeline or strategy logic.
- Adding new regime states without rewriting existing strategy templates.
- Adding new feature sources (new AMT tool, new macro indicator) without changing the classifier architecture.
- Running on new instruments or asset classes by providing new data and retraining.

### 11.2 Regime State Flexibility

The 6-state model is a starting point. The architecture supports:

- **Collapsing states:** if "Ranging After Uptrend" and "Ranging After Downtrend" prove statistically indistinguishable, collapse them into a single "Ranging" state. The system becomes simpler without architectural change.
- **Expanding states:** if analysis reveals a meaningful sub-state (e.g., "volatile trending" vs "calm trending"), add it. Requires new labels, a strategy template, and retraining — but no pipeline changes.
- **Per-instrument state counts:** equities may exhibit 6 distinct regimes while FX only shows 4. The classifier can be configured per asset class.

### 11.3 Strategy Adaptability

Per-regime strategies are parameterised templates, not hardcoded logic. This supports:

- **Parameter reoptimisation:** Optuna reruns quarterly on recent data to adapt parameters to evolving market conditions.
- **Strategy replacement:** if a better mean-reversion approach is discovered for ranging markets, swap it into the template without touching the classifier or other regime strategies.
- **New strategy overlays:** add options-based strategies for transition regimes, pair trading for ranging regimes, or cross-asset hedging — each as a new template.

### 11.4 Degradation Handling

The system should degrade gracefully when components underperform:

- **Low classifier confidence:** reduce position size, widen stops, or go flat. Do not force a regime assignment when uncertain.
- **Strategy underperformance:** if a regime-strategy pair's rolling Sortino drops below 1.0, reduce allocation to that strategy and flag for review.
- **Data unavailability:** fall back to Pomorski-only features if AMT data is missing. Fall back to no-trade if daily data is missing.
- **Correlation spike:** if cross-instrument correlations exceed 0.8, reduce aggregate exposure regardless of individual regime classifications (systemic risk override).

---

## 12. Development Phases

### Phase 1: Foundation (Single instrument, daily frequency, Pomorski features only)

**Scope:** SPY (or ES futures) only. Daily OHLCV + VIX + yields. Rule-based labeller (v1). 3-class classifier (up/down/sideways). Single strategy per regime (long/short/flat). Walk-forward backtest 2010-2024.

**Success criteria:** MCC > 0.3 on 3-class problem. Positive alpha after costs in factor regression.

**Deliverables:** working end-to-end pipeline, baseline performance metrics, initial SHAP analysis.

### Phase 2: AMT Integration (Single instrument, AMT features added)

**Scope:** add intraday data for AMT feature computation (TPO, volume profile, VWAP, bar-level delta). Expand to 6-class problem. Compare classifier performance with and without AMT features.

**Success criteria:** AMT features add ≥2 PCA dimensions. MCC improvement ≥ 0.05 over Phase 1. At least one AMT feature in top 10 SHAP importance.

**Deliverables:** AMT feature engine, 6-class classifier, PCA comparison analysis.

### Phase 3: Strategy Optimisation (Per-regime strategy templates)

**Scope:** implement parameterised strategy templates per regime. Optimise parameters via Optuna on walk-forward splits. Implement adaptive stops at AMT structural levels. Full attribution analysis (Tests A/B/C/D).

**Success criteria:** Sortino > 2.0 after costs. Alpha significant in Fama-French regression. Each regime-strategy pair independently profitable.

**Deliverables:** strategy templates, full backtest results, factor regression analysis, attribution test results.

### Phase 4: Multi-Asset Expansion

**Scope:** extend to equities (5-10 instruments), commodities (5-10 futures), and optionally FX. Per-asset-class classifiers. Portfolio-level risk management. Correlation-aware sizing.

**Success criteria:** alpha persists across asset classes. Portfolio Sortino > 2.0. Maximum drawdown < 25%.

**Deliverables:** multi-asset pipeline, portfolio construction module, cross-asset analysis.

### Phase 5: Live Validation

**Scope:** paper trading for 3-6 months. Broker API integration. Daily automated execution. Performance monitoring dashboard.

**Success criteria:** forward alpha within 50% of backtested alpha. No catastrophic failures. System runs unattended daily.

**Deliverables:** live trading bot, monitoring dashboard, forward performance report.

---

## 13. Key References

- Pomorski, P. (2024). *Construction of Effective Regime-Switching Portfolios Using a Combination of Machine Learning and Traditional Approaches.* PhD thesis, University College London.
- Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.
- Kaufman, P.J. (1995). *Smarter Trading: Improving Performance in Changing Markets.* McGraw-Hill.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Boyd, S. et al. (2017). Multi-period trading via convex optimization. *Foundations and Trends in Optimization*, 3(1), 1-76.
- Ang, A. & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.
- Wyckoff, R.D. (1937). *The Richard D. Wyckoff Method of Trading and Investing in Stocks.*
- Dalton, J.F. (1993). *Mind Over Markets: Power Trading with Market Generated Information.*
- Steidlmayer, J.P. (1986). *Markets and Market Logic.* Porcupine Press.
