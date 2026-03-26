# AMT-Regime-Quantitative-Algorithm

This project builds a quantitative trading system that decomposes financial markets into discrete structural regimes using Auction Market Theory (AMT) and deploys optimised strategies per regime. The core thesis is that markets cycle through identifiable behavioural states — trending, ranging, and transitional — and that strategies tuned to each state will outperform static approaches.

The architecture extends the work of Pomorski (2024), *"Construction of Effective Regime-Switching Portfolios Using a Combination of Machine Learning and Traditional Approaches"* (UCL PhD thesis), which validated a detection-prediction-optimisation pipeline using KAMA+MSR for regime labelling, Random Forest for regime prediction, and Model Predictive Control for portfolio construction. This project diverges from Pomorski by:

- Expanding from 4 regimes (volatility × trend) to 6 regimes incorporating directional context and AMT structural states.
- Grounding regime definitions in Auction Market Theory rather than pure statistical volatility decomposition.
- Integrating market microstructure data (TPO, volume profile, delta, VWAP, open interest) as classifier features alongside traditional momentum and volatility indicators.
- Deploying distinct, optimised strategy templates per regime rather than a single long/short approach.
- Using AMT structural levels (value area high/low, POC) for execution-level risk management, including adaptive stop losses that act as real-time regime change detectors.

The predicted edge is structural alpha from two sources: (1) improved regime transition detection via AMT microstructure features that capture participant behaviour, not just price action, and (2) per-regime strategy optimisation that exploits the distinct statistical properties of each market state.