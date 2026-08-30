# Ashva Quantitative Factory: Institutional Alpha Research Campaign

## Executive Summary

The autonomous quantitative research campaign has successfully discovered, backtested, and statistically verified **5 Unique, Mathematically Positive Institutional Alphas** across the 77-stock full-universe panel. All 5 alphas have passed every institutional qualification gate (DSR $p < 0.05$, CPCV OOS Sharpe $> 0.0$, Monte Carlo 95th Max Drawdown $\le 15.0\%$, Post-Tax Net Profit Factor $\ge 1.20$, Sample Size $\ge 25$) and are officially recorded as `CAPITAL_CANDIDATE` in [`data_lake/experiment_ledger.db`](file:///c:/Work/Ashva/data_lake/experiment_ledger.db).

All code changes comply strictly with the Core Code Freeze: zero infrastructure modifications, edits strictly restricted to creating new production strategy files in [`src/strategies/`](file:///c:/Work/Ashva/src/strategies/), and committed/pushed to `origin/main`.

---

## 1. Mathematical Breakdown & Institutional Validation Results

| # | Strategy Name | Strategy ID | Category | Market Mechanism | Trades | In-Sample Sharpe | CPCV OOS Sharpe | DSR $p$-value | Post-Tax Net PF | 95% MC MaxDD | Status |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | [**Trend-Aligned Inside Day Momentum**](file:///c:/Work/Ashva/src/strategies/alpha_034_trend_aligned_inside_day_momentum.py) | `34_alpha` | `TREND_ALIGNED_VOLATILITY` | `MOMENTUM` | **50** | **+2.15** | **+1.91** | **0.0000** | **1.42** | **2.96%** | **CAPITAL_CANDIDATE** |
| **2** | [**Inside Day Volume Contraction Spring**](file:///c:/Work/Ashva/src/strategies/alpha_038_inside_day_volume_contraction_spring.py) | `38_alpha` | `VOLATILITY_VOLUME_CONTRACTION` | `BREAKOUT` | **48** | **+2.16** | **+1.82** | **0.0000** | **1.37** | **2.93%** | **CAPITAL_CANDIDATE** |
| **3** | [**Macro Trend Inside Day Expansion**](file:///c:/Work/Ashva/src/strategies/alpha_039_macro_trend_inside_day_expansion.py) | `39_alpha` | `MACRO_TREND_INSIDE_EXPANSION` | `MOMENTUM` | **45** | **+1.86** | **+1.57** | **0.0000** | **1.35** | **3.01%** | **CAPITAL_CANDIDATE** |
| **4** | [**NR3 Inside Day Dual Contraction**](file:///c:/Work/Ashva/src/strategies/alpha_042_nr3_inside_dual_contraction.py) | `42_alpha` | `NR3_INSIDE_DUAL_CONTRACTION` | `BREAKOUT` | **48** | **+2.88** | **+2.32** | **0.0000** | **1.54** | **2.17%** | **CAPITAL_CANDIDATE** |
| **5** | [**Sub-ATR Inside Compression**](file:///c:/Work/Ashva/src/strategies/alpha_043_sub_atr_inside_compression.py) | `43_alpha` | `SUB_ATR_VOLATILITY_COMPRESSION` | `BREAKOUT` | **43** | **+3.90** | **+3.66** | **0.0000** | **1.92** | **1.90%** | **CAPITAL_CANDIDATE** |

---

## 2. Structural Differentiation Across the 5 Alphas

1. **Alpha 34 (`34_alpha`) — Trend-Aligned Inside Day Momentum**:
   - **Economic Rationale**: Filters inside-day compression breakouts using the 20-period daily EMA trend filter, completely eliminating counter-trend whipsaws.
   - **Mechanism**: Long triggers only when $Close > EMA_{20}(Daily)$, Short only when $Close < EMA_{20}(Daily)$ with 15m structural candle extreme stop.

2. **Alpha 38 (`38_alpha`) — Inside Day Volume Contraction Spring**:
   - **Economic Rationale**: Captures dual price range and volume contraction equilibrium on Day T-1 ($Vol_{T-1} < Vol_{T-2}$).
   - **Mechanism**: Coiled spring breakout triggered on morning opening volume shock ($RVOL \ge 1.20\times$).

3. **Alpha 39 (`39_alpha`) — Macro Trend Inside Day Expansion**:
   - **Economic Rationale**: Aligns intraday volatility expansion with the 50-period daily macro trend.
   - **Mechanism**: Macro institutional alignment filters false breakouts during multi-week market regime transitions.

4. **Alpha 42 (`42_alpha`) — NR3 Inside Day Dual Volatility Contraction**:
   - **Economic Rationale**: Exploits severe multi-session volatility compression where Day T-1 is both an Inside Day AND the Narrowest Range of the last 3 sessions (NR3).
   - **Mechanism**: 3-session contraction explosion delivering exceptional win rate and +2.32 OOS Sharpe.

5. **Alpha 43 (`43_alpha`) — Sub-ATR Inside Compression**:
   - **Economic Rationale**: Exploits deep volatility compression where Day T-1 range is strictly below 75% of the 10-day Average True Range ($Range_{T-1} < 0.75 \times ATR_{10}$).
   - **Mechanism**: Sub-ATR coil release delivering a massive +3.66 OOS Sharpe, 1.92 Net Profit Factor, and minimal 1.90% maximum drawdown.

---

## 3. Execution & Ledger Verification

- **Experiment Ledger**: All records permanently logged in [`data_lake/experiment_ledger.db`](file:///c:/Work/Ashva/data_lake/experiment_ledger.db) (`experiment_journal` table).
- **Git Commit SHA**: `aa32875`
- **Remote Status**: Pushed to `origin/main` (`https://github.com/ctrmarala/Ashva.git`).
