# Ashva Institutional Quantitative Alpha Research Principles

The following 5 principles are strictly enforced across the Ashva Alpha Factory. No agent, subagent, or developer may bypass them:

1. **Zero Ad-Hoc Runners**:
   - Never write one-off research runner scripts (e.g. `research_alpha_X.py`).
   - ALL alpha backtesting, evaluation, and qualification must execute exclusively through `scripts/research_alpha.py`.

2. **Full Dynamic Universe (77 Stocks)**:
   - Never substitute an 8-stock or sub-sample panel for full-universe evaluation or timeframe discovery.
   - All 77 active equities from `get_universe_symbols()` must be evaluated.

3. **Algorithmic Timeframe Selection**:
   - Never hardcode a preferred timeframe.
   - The preferred timeframe must be discovered empirically by the quantitative composite scoring function in `research_alpha.py`.

4. **Dynamic Real-Market Regime Engine**:
   - Never classify regimes using arbitrary calendar date intervals.
   - All trades must be classified into `BULL`, `BEAR`, or `FLAT` using point-in-time benchmark moving-average and trend-slope market structure.

5. **True Panel-Level Statistical Qualification**:
   - Deflated Sharpe Ratio (DSR), Combinatorial Purged Cross-Validation (CPCV), and 5,000 Monte Carlo tail simulations must be calculated on the combined 77-symbol panel return series.
   - All strategy code must pass `AlphaLinter` with zero lookahead leakage, dynamic universe binding, and 15:15 IST square-off.