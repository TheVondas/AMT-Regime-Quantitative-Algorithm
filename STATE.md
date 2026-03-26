# Current Project State

**Last updated:** 2026-03-26

## Current Stage
Stage 1, Week 0 (pre-development — planning complete)

## What's Next
- [ ] Week 1: Environment setup, data acquisition (SPY OHLCV, VIX, yields), data cleaning pipeline

## Active Concerns
- Intraday data source for Stage 2 AMT features: Polygon.io vs Alpaca vs Databento. Decision needed by Week 7
- Whether to include short strategies or go long-only for v1
- Futures (ES) vs ETF (SPY) as primary instrument

## Known Issues
- None yet (pre-development)

---

# Decision Log (append-only — never delete entries, only add)

### 2026-03-26 — Project inception
- **Classifier:** Random Forest (validated by Pomorski, interpretable via SHAP)
- **Labeller v1:** Rule-based (KAMA/EMA trend + ATR volatility + prior state context). KAMA+MSR is v2 upgrade
- **Starting instrument:** SPY, daily frequency
- **Starting features:** Pomorski-style only. AMT features added in Stage 2
- **Latency target:** End of day. No intraday/HFT
- **Transaction costs:** 0.2% round-trip baseline, sensitivity tested at 0.1-0.8%
- **Alpha definition:** Significant intercept in Fama-French 6-factor regression after costs
- **Stops:** Broker-side orders at AMT structural levels, not algorithm-managed
- **6 regimes:** Trending Up, Trending Down, Ranging Neutral, Ranging After Uptrend (Distribution), Ranging After Downtrend (Accumulation), Transition/Breakout
- **Architecture:** Features → RF classifier → regime probability vector → per-regime strategy → position sizing → execution
- **Transition handling:** Soft probability thresholds + probability rate of change + circuit breaker + AMT structural stops
- **Context management:** Use STATE.md for session continuity. Read before every edit. Decision log is append-only

---

# Session Log (append-only — one entry per working session)

### 2026-03-26 — Session 1: Planning
**What was done:**
- Analysed Pomorski (2024) UCL thesis in full — extracted methodology, features, hyperparameters, results
- Designed 6-state regime model grounded in AMT
- Discussed feasibility, latency, data sourcing, AMT feature integration, alpha measurement, overfitting risks
- Wrote PROJECT_PLAN.md (strategic overview)
- Wrote DEVELOPMENT_PLAN.md (week-by-week execution with checklists)
- Set up memory files for cross-session persistence

**Key takeaways:**
- Alpha comes from per-regime strategy optimisation, not regime detection itself (detection is infrastructure)
- AMT microstructure features (TPO, VP, delta, VWAP) capture participant behaviour that price-derived features miss — especially at regime transitions
- Adaptive stops at AMT structural levels (VAH/VAL) act as execution-level regime change detectors independent of the classifier
- Pomorski's RF achieved MCC ~0.43, Sortino 4.7-25.0 — sets the benchmark to match or beat
