# Ashva Factory v2.3 Contract Gap Resolution -- Architectural Design Document

**Date**: 2026-08-19  
**Status**: DESIGN ONLY (Autonomous Discovery HALTED -- Zero Implementation Code)  
**Objective**: Resolve the Cross-Sectional Selector and Multi-Day Swing capability boundaries with the smallest possible architectural extension, preserving 100% of the frozen Factory v1 infrastructure.

---

## 1. Executive Summary & Problem Formulation

The adversarial audit of Alpha 32 and Alpha 33 revealed that **the discovery controller was conceptually proposing cross-sectional selection and multi-day swing holding, while the underlying execution contract could only execute single-asset intraday strategies**.

```text
+---------------------------------------------------------------------------------------+
| CORE CAPABILITY & CONTRACT BOUNDARIES                                                 |
+-------------------------------------------------------+-------------------------------+
| Strategy Archetype                                    | Execution Contract            |
+-------------------------------------------------------+-------------------------------+
| Type A: Single-Asset Signal Alpha                     | DataFrame -> Signals          |
|         (e.g. Alphas 01-31)                           | (FROZEN & SUPPORTED)          |
|                                                       |                               |
| Type B: Cross-Sectional Selector Alpha                | Dict[str, DataFrame]          |
|         (e.g. Dynamic Top-3 RVOL / RS Selection)      | -> Dynamic Subsets            |
|                                                       | (CONTRACT GAP)                |
|                                                       |                               |
| Type C: Multi-Day Swing Delivery Alpha                | Multi-Session Margin          |
|         (e.g. 2-to-5 Day Delivery Holding)            | & Overnight Fills             |
|                                                       | (DEFERRED TO v3.0)            |
+-------------------------------------------------------+-------------------------------+
```

The goal of Factory v2.3 is to **solve Type B cleanly without modifying BaseHypothesis, BacktestEngine, or IndianCostModel**, while explicitly deferring Type C.

---

## 2. Answers to the 9 Core Architectural Questions

### Q1: Can selector alphas be implemented as a thin portfolio-selection layer without changing BaseHypothesis?
**YES, absolutely.**
A Selector is mathematically a **point-in-time universe filter/mask**, not a directional entry signal generator:
- `BaseSelector`: Operates on `Dict[str, DataFrame]` at decision time `t` (09:15 IST), computes contemporaneous cross-sectional metrics across all assets, and outputs a boolean eligibility mask `Dict[str, Series[bool]]`.
- `BaseHypothesis`: Remains **100% untouched**. It continues to take a single stock's `DataFrame` and generate its standard `Signals: Series[float]` (+1, -1, 0).
- **The Execution Composition**:
  `Effective Signal[i, t] = Selector Mask[i, t] * Hypothesis Signal[i, t]`
This separates **Asset Selection** from **Signal Generation** with zero coupling.

---

### Q2: Can the existing BacktestEngine execute selected symbols without becoming selector-aware?
**YES.**
Because the signal composition occurs upstream:
- The controller applies the selector mask before passing each asset's signals to `BacktestEngine.run()`.
- For unselected assets on day `d`, `signals = 0.0` at all bars.
- `BacktestEngine` runs its exact frozen execution loop, fills, slippage, and statutory tax calculations on each asset without needing any selector awareness.
- **Zero changes are required to BacktestEngine, execution logic, or IndianCostModel**.

---

### Q3: What is the smallest interface required?
A single, lightweight abstract base class in a dedicated module `src/research/selector.py` (~25 lines of Python):

```python
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class BaseSelector(ABC):
    """
    Abstract Base Class for Cross-Sectional Universe Selectors.
    Evaluates multi-asset contemporaneous data at decision time t
    and produces point-in-time eligibility masks.
    """

    @abstractmethod
    def select_universe(self, universe_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Parameters:
            universe_data: Dict mapping symbol -> contemporaneous OHLCV DataFrame.
        Returns:
            Dict mapping symbol -> boolean Series (True = eligible to trade, False = blocked).
        """
        pass
```

---

### Q4: Which existing components must change, if any?
- **Modified Components**: **NONE** of the core factory modules change.
- **New Additions**:
  1. `src/research/selector.py`: Defines `BaseSelector` and built-in selectors (e.g. `TopKRelativeVolumeSelector`, `SectorMembershipSelector`, `PassThroughSelector`).
  2. `src/research/discovery_controller.py`: Updated in v2.3 to optionally bind a `BaseSelector` to a candidate during research orchestration.
  3. `tests/test_selector_contracts.py`: Dedicated contract test suite for cross-sectional point-in-time integrity.

---

### Q5: Which components must remain FROZEN?
The following core infrastructure remains **100% frozen and unmodified**:
- `src/research/hypothesis.py` (`BaseHypothesis`, `HypothesisMetadata`, `StrategyHorizon`, `MarketMechanism`)
- `src/backtest/engine.py` (`BacktestEngine`)
- `src/backtest/cost_model.py` (`IndianCostModel`)
- `src/data/data_lake.py` (`DataLake`)
- `src/research/validator.py` (`StatisticalValidator`)
- All 31 existing strategy implementations (`src/strategies/alpha_03.py` through `alpha_31.py`).

---

### Q6: How will selector-specific contract tests prevent another Alpha-33-type discrepancy?
A new test suite `tests/test_selector_contracts.py` will enforce three mathematical invariants:
1. **Strict Cardinality Constraint**:
   For any timestamp `t`, the number of selected assets across the universe must never exceed `k`:
   `sum(Mask[s, t] == True for s in Universe) <= k`
2. **Zero-Execution Invariant on Unselected Assets**:
   Assert that when `Mask[s, t] == False`, `BacktestEngine.run()` produces exactly 0 trades and 0.0 PnL for asset `s`.
3. **Difference Assertion**:
   Assert that a strategy evaluated with a Top-3 selector produces strictly fewer trades than the unconditional per-stock threshold strategy, proving it is not running independent per-stock filters.

---

### Q7: How will we test point-in-time correctness?
1. **Future Bar Mutation Test**:
   - For any asset `B`, mutate future prices/volumes at `t+1 ... T` (injecting extreme spikes).
   - Assert that the selector ranking and selection decision for asset `A` at timestamp `t` is **100% bit-for-bit identical**.
2. **Contemporaneous Timestamp Alignment Test**:
   - Assert that calculating the metric at timestamp `t` (09:15 IST) uses strictly data with index `<= t`.

---

### Q8: How will the existing 31 alphas remain completely unaffected?
- Single-stock signal alphas (Alphas 01-31) do not declare a selector (or implicitly use `PassThroughSelector`, where `Mask[s, t] = True` everywhere).
- Their existing test suite (`tests/test_strategy_contracts.py`, 30/30 passed) runs unmodified against the frozen `BaseHypothesis` contract.

---

### Q9: Is multi-day trading actually needed now, or should it remain future scope?
- **Verdict: DEFERRED TO FUTURE SCOPE (Factory v3.0)**.
- **Decision Rationale**:
  - Intraday equities with mandatory 15:15 IST square-off is the proven, high-frequency focus of the Ashva project.
  - Multi-day swing delivery introduces major operational complexities:
    1. Overnight margin requirements (100% cash vs 5x intraday leverage).
    2. Higher statutory delivery STT (0.1% on buy + 0.1% on sell vs 0.025% sell-side intraday).
    3. Overnight gap risk models and corporate action adjustments.
  - Implementing multi-day swing trading now would create significant infrastructure churn for zero immediate intraday gain.
  - Therefore, multi-day swing trading is explicitly deferred to **Factory v3.0**.

---

## 3. High-Level Architectural Flowchart (Factory v2.3)

```text
+---------------------------------------------------------------------------------------+
| FACTORY v2.3 CLEAN TWO-STAGE PIPELINE                                                 |
+---------------------------------------------------------------------------------------+
|                                                                                       |
| [STAGE 1: ASSET SELECTION] (Optional Cross-Sectional Layer)                           |
|                                                                                       |
|   Full Universe DataLake OHLCV (Dict[symbol, DataFrame])                              |
|                        |                                                              |
|                        v                                                              |
|               BaseSelector.select_universe()                                          |
|               (Ranks cross-section at t = 09:15 IST)                                  |
|                        |                                                              |
|                        v                                                              |
|   Point-in-Time Eligibility Mask (Dict[symbol, Series[bool]])                         |
|                        |                                                              |
| -----------------------+------------------------------------------------------------- |
|                        v                                                              |
| [STAGE 2: SIGNAL GENERATION & EXECUTION] (Frozen v1 Layer)                            |
|                                                                                       |
|   BaseHypothesis.generate_signals(single_asset_df)                                    |
|                        |                                                              |
|                        v                                                              |
|   Raw Signals (Series[float])                                                         |
|                        |                                                              |
|                        v                                                              |
|   Effective Signals = Mask * Raw Signals                                              |
|                        |                                                              |
|                        v                                                              |
|   BacktestEngine.run(strategy, data, effective_signals) [FROZEN]                      |
|                        |                                                              |
|                        v                                                              |
|   IndianCostModel (STT, Stamp Duty, GST, Turnover Fees) [FROZEN]                      |
|                        |                                                              |
|                        v                                                              |
|   StatisticalValidator (CPCV, Monte Carlo, Live SQLite DSR) [FROZEN]                  |
|                                                                                       |
+---------------------------------------------------------------------------------------+
```

---

## 4. Summary & Freeze Recommendation

1. **Architecture Status**: Clean, decoupled, and minimal.
2. **Factory v1 Infrastructure**: **100% FROZEN & UNMODIFIED**.
3. **Next Steps**: Awaiting your review and explicit approval of `factory_v23_contract_design.md` before writing any implementation code or re-enabling autonomous discovery.
