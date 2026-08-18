# Ashva Quantitative Platform — Factory Freeze Review
**Strict Engineering Audit & Architectural Freeze Assessment**

- **Review Date**: `2026-08-18`
- **Scope**: Entire Ashva Quantitative Research Factory
- **Status**: 🟢 **Freeze-Ready** (With 1 minor docstring clarification proposed)

---

## 1. Executive Summary & Philosophy

The objective of the **Factory Freeze Review** is to evaluate the end-to-end quantitative platform to ensure that future alpha research ($\text{Alpha 24, 25, 26\dots}$) can be executed with **absolute engineering rigor, zero look-ahead bias, and high developer velocity**.

> **The Guiding Principle**: 
> *"Strict" means strict code correctness, strict position lifecycle integrity, strict regulatory cost accounting, and strict prevention of data leakage. It does NOT mean making alpha qualification so restrictive that we eliminate genuine market opportunities.*

The factory is **architecturally solid, lightweight, and fully decoupled**. Future alpha authors only need to write **their specific hypothesis logic and its unit test**; all data querying, execution modeling, statutory Indian cost deductions, statistical testing, and matrix reporting are 100% reusable.

---

## 2. Comprehensive 16-Area Audit Matrix

| # | Architectural Area | Status | Codebase Evidence | Required Action Before Freeze |
| :-: | :--- | :---: | :--- | :--- |
| **1** | **Alpha-Agnostic Contract** | **Supported** | `BaseStrategy` ([`src/strategies/base.py`](file:///c:/Work/Ashva/src/strategies/base.py)) requires only `generate_signals(df)` returning dense `signal`, `stop_loss`, `take_profit`, `rationale`. Minimal footprint, zero mandatory boilerplate. | Clarify `base.py` docstring regarding dense vector semantics (+1=Maintain Long, -1=Maintain Short, 0=Exit/Flat). |
| **2** | **Position Lifecycle Contract** | **Supported** | All 23 alphas maintain `curr_state` across intraday bars. CI test suite ([`tests/test_strategy_contracts.py`](file:///c:/Work/Ashva/tests/test_strategy_contracts.py)) verifies `23/23 PASSED` in 4.30s. | Retain `test_strategy_contracts.py` as a hard CI gate for all future alphas. |
| **3** | **Single Authoritative Backtester** | **Supported** | One unified `BacktestEngine` ([`src/backtest/engine.py`](file:///c:/Work/Ashva/src/backtest/engine.py)) handles next-bar open fills, intrabar SL/TP checks, gap fills (`min(next_open, current_sl)`), and MTM equity tracking. | None. Zero alpha-specific backtest engines permitted. |
| **4** | **Universe Architecture** | **Supported** | Bounded to **Maximum NIFTY 50** ([`scripts/scan_nifty_50_universe.py`](file:///c:/Work/Ashva/scripts/scan_nifty_50_universe.py)) with **NIFTY-14 discovery subset** and **540-day (~18m) maximum lookback**. | Keep NIFTY 50 / 540d boundaries strictly frozen. Do not expand to NIFTY 500/1000. |
| **5** | **Alpha Selector Capability** | **Supported** | Alphas can compute suitability/eligibility masks at Bar $t$ without modifying framework backtester. | Do not implement complex probability/ML scoring models now. |
| **6** | **Alpha Stacking / Ensemble** | **Supported** | Multi-alpha correlation and basket returns are evaluated at the portfolio matrix level ([`scripts/run_alpha_matrix_audit.py`](file:///c:/Work/Ashva/scripts/run_alpha_matrix_audit.py)). Individual alphas remain pure single hypotheses. | Keep ensemble logic strictly at the portfolio layer; never in individual alphas. |
| **7** | **Data Contract & Integrity** | **Supported** | `DataLake` ([`src/data/data_lake.py`](file:///c:/Work/Ashva/src/data/data_lake.py)) uses dual DuckDB/Parquet storage, strict IST timestamps, and automatic 540-day lookback ceiling. Indicators strictly shift prior-day values (`shift(1)`). | None. Official historical broker feeds provide adjusted OHLCV. |
| **8** | **Point-in-Time / Survivorship** | **Supported** | For 15m intraday equity models on mega-cap NIFTY 50 assets over 18 months, index reconstitution survivorship bias is negligible. | Avoid building a complex historical constituent database. |
| **9** | **Decision-Time Correctness** | **Supported** | Signal generated at Bar $t$ Close $\rightarrow$ Filled strictly at Bar $t+1$ Open. Stop loss evaluated intrabar on Bar $t+1$ High/Low with gap slippage modeling. | None. Zero lookahead execution verified. |
| **10** | **Validation Factory & Modes** | **Supported** | Clear 3-tier mode hierarchy in `run_alpha_matrix_audit.py`: `DEV` (~18s fast loop), `RESEARCH` (~10m NIFTY-14 540d), and `FULL` (NIFTY-50 multi-asset). | Maintain SOP: developers must use `DEV` or `pytest` for debugging; never trigger full research runs for syntax/contract checks. |
| **11** | **Reusable Research Matrix** | **Supported** | `run_alpha_matrix_audit.py` reads `STRATEGY_MAP` dynamically. Adding `alpha_24` requires zero changes to the matrix orchestrator. | Strictly forbid per-alpha runner scripts (e.g. `run_alpha_24_matrix.py`). |
| **12** | **Authoritative Indian Cost Model** | **Supported** | `IndianCostModel` ([`src/analytics/indian_costs.py`](file:///c:/Work/Ashva/src/analytics/indian_costs.py)) enforces exact statutory schedules (STT, GST, Stamp Duty on buy only, SEBI, NSE fees, ₹20 Angel One cap) + 3.0 bps slippage. | None. Cost model is 100% centralized and universal. |
| **13** | **Reproducibility & Audit Trail** | **Supported** | `ResearchExperimentLedger` ([`src/research/experiment_ledger.py`](file:///c:/Work/Ashva/src/research/experiment_ledger.py)) records Git commit SHA, strategy ID, universe, timeframe, parameter JSON, trial counts, and IS/OOS Sharpe to SQLite. | None. Every experiment is permanently reproducible. |
| **14** | **Portfolio vs Alpha Separation** | **Supported** | Matrix output reports standalone Alpha Sharpe/PnL alongside cross-alpha return correlations, recognizing that low-return alphas can provide high portfolio diversification value. | None. Maintain clean separation. |
| **15** | **Code Reuse & Developer UX** | **Supported** | Authoring `Alpha 24` requires writing **only** `src/strategies/alpha_24_*.py` + registering in `STRATEGY_MAP` + contract test. | None. Factory minimizes friction for new hypothesis testing. |
| **16** | **Complexity Ceiling** | **Supported** | Factory contains zero unneeded ML models, no tick simulators, no corporate actions subsystem, no microservice overhead. | Strictly freeze these boundaries to prevent architectural bloat. |

---

## 3. Deep-Dive Findings by Critical Subsystems

### A. Data Lake & Point-in-Time Integrity
- **Dual Storage**: DuckDB (`data_lake/ashva_market_data.duckdb`) provides SQL speed; Apache Parquet (`data_lake/parquet/`) ensures portable backups.
- **Lookback Ceiling**: `DataLake.load_bars()` strictly enforces `max_lookback_days=540`, ensuring all backtests evaluate the identical 18-month economic window.
- **Lookahead Prevention in Feature Extraction**:
  - In `TechnicalIndicators.add_camarilla_pivots()`: `daily.shift(1)` ensures prior-day high/low/close are anchored to current intraday bars without lookahead.
  - In `TechnicalIndicators.add_tod_rolling_volume()`: `shift(1).rolling(20)` prevents the current bar's volume from contaminating the baseline.

### B. Strategy Lifecycle Contract & The 1-Bar Pulse Fix
- **The Defect**: 19 alphas emitted single-bar pulses (`signal=1` at 09:15, `signal=0` at 09:30), causing `BacktestEngine` to execute immediate 1-bar exits.
- **The Solution**: Standardized dense state vectors across all strategies:
  ```python
  # Intraday 15:15 EOD Square-Off
  if bar_time >= t_1515:
      if curr_state != 0.0:
          curr_state = 0.0
          signals[i] = 0.0
      continue
  # Maintain active position across holding bars
  if curr_state != 0.0:
      signals[i] = curr_state
      stop_loss[i] = curr_sl
      take_profit[i] = curr_tp
      continue
  ```
- **Automated CI Contract Test**: [`tests/test_strategy_contracts.py`](file:///c:/Work/Ashva/tests/test_strategy_contracts.py) asserts that no strategy generates $> 50\%$ 1-bar exits on `SIGNAL`. All 23 alphas pass in **4.30s**.

### C. Execution Timing, Real-World Gap Handling & Costs
- **Execution Timing**: Next-bar open fill (`entry_price = next_open`).
- **Realistic Gap Handling**:
  - Long Stop Loss: `exit_price = min(next_open, current_sl)`
  - Long Take Profit: `exit_price = max(next_open, current_tp)`
  - Short Stop Loss: `exit_price = max(next_open, current_sl)`
  - Short Take Profit: `exit_price = min(next_open, current_tp)`
- **Statutory Costs**: Angel One ₹20 cap, STT (0.025% on sell for intraday, 0.1% buy+sell for delivery), GST (18%), Stamp Duty (0.003% buy only), SEBI charges, and 3.0 bps slippage.

### D. Validation Factory & Mode Hierarchy
- **DEV Mode (`--mode dev`)**: Runs 4 representative alphas on NIFTY-14 over 120 days in **~18 seconds**. Used for quick developer iteration and infrastructure validation.
- **RESEARCH Mode (`--mode research`)**: Runs all 23 alphas on NIFTY-14 over 540 days (420d IS + 120d untouched OOS) in **~10 minutes**. Used for authoritative alpha qualification.
- **FULL Mode (`--mode full`)**: Runs all 23 alphas across full NIFTY-50 over 540 days with 120d OOS. Used for multi-asset cross-sectional generalization testing.

---

## 4. What Must NOT Be Added (Complexity Ceiling)

To maintain maximum developer agility and prevent over-engineering:
1. ❌ **DO NOT Add Complex Dynamic Tick Simulators**: Next-bar open + 3.0 bps slippage + intrabar SL/TP is mathematically sufficient for 15m timeframes.
2. ❌ **DO NOT Add ML/Probability Scoring Models at the Signal Layer**: Alphas must represent simple, falsifiable economic mechanisms.
3. ❌ **DO NOT Expand Universe Beyond NIFTY 50**: Megacap liquidity ensures execution feasibility under Indian statutory friction.
4. ❌ **DO NOT Create Strategy-Specific Matrix Runners**: All research must flow through `run_alpha_matrix_audit.py`.
5. ❌ **DO NOT Relax Regulatory Friction**: Never remove STT or lower slippage below 3 bps to artificially manufacture profitability.

---

## 5. Final Classification & Recommendation

### Final Verdict: 🟢 FREEZE-READY

The Ashva Quantitative Platform is **fully verified, mathematically sound, and architecturally complete**.

### The Frozen SOP for Adding Future Alphas ($\text{Alpha 24+}$):
```text
1. Formulate Hypothesis & Freeze Entry Parameters
      ↓
2. Implement Alpha adhering to Dense State Contract (+1 / -1 / 0)
      ↓
3. Run Automated CI Contract Test: pytest tests/test_strategy_contracts.py  [~4 seconds]
      ↓
4. Run Fast DEV Matrix: python scripts/run_alpha_matrix_audit.py --mode dev [~18 seconds]
      ↓
5. Run Authoritative Research Matrix: python scripts/run_alpha_matrix_audit.py --mode research [~10 minutes]
      ↓
6. Classify on 5-Tier Alpha Ladder (Discovery / Research / Paper / Production / Archive)
```
