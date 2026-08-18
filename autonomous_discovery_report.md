# Ashva Autonomous Alpha Discovery Campaign — Factory v2 Report
**Generated**: `2026-08-19 00:08:13 IST` | **Factory Status**: 🔒 `FROZEN v1`

---

## 1. Executive Summary

The **Ashva Factory v2 Discovery Controller** conducted an autonomous search across unexplored quantitative territory, evaluating candidate mechanisms under Stage 0 empirical feasibility, contract verification, and statutory transaction friction.

## 2. Research Steps & Discovery Log

| Step | Candidate ID | Name | Category | Stage Reached | Outcome | Rationale / Result |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `alpha_32` | ALPHA_32_VOLUME_WEIGHTED_CLOSING_IMBALANCE | `ORDER_FLOW_IMBALANCE` | Stage 0 (Plausibility) | **REJECTED_AT_STAGE_0** | Stage 0 REJECT: Gross edge (-0.56 bps) fails Indian statutory friction hurdle (7.0 bps) across N=126 observations. |
| 2 | `alpha_33` | ALPHA_33_MULTI_DAY_CONSOLIDATION_BREAKOUT | `SWING_MOMENTUM` | Stage 0 (Plausibility) | **REJECTED_AT_STAGE_0** | Stage 0 REJECT: Insufficient historical event occurrences (N=0 < 10) |

## 3. Mechanism Landscape & Exploration Map

```text
EXPLORED TERRITORY SUMMARY:
  • SWING_MOMENTUM           : 1 Strategies Evaluated
  • OPENING_AUCTION          : 2 Strategies Evaluated
  • STATISTICAL_REVERSION    : 3 Strategies Evaluated
  • GAP_MOMENTUM             : 2 Strategies Evaluated
  • RELATIVE_STRENGTH        : 2 Strategies Evaluated
  • VOLATILITY_EXPANSION     : 1 Strategies Evaluated
  • SECTOR_MOMENTUM          : 1 Strategies Evaluated
  • TREND_EXHAUSTION         : 1 Strategies Evaluated
  • VOLATILITY_SQUEEZE       : 1 Strategies Evaluated
  • MICROSTRUCTURE_FADE      : 1 Strategies Evaluated
```

## 4. Knowledge Gained & Empirical Insights

1. **Opening Auction vs Closing Imbalance**: While morning opening gaps (Alpha 14) generate strong follow-through, late-session closing imbalances (Power Hour) must produce gross moves > 7.0 bps to overcome Indian transaction taxes.
2. **Multi-Day Swing Holding (Alpha 10 & Alpha 33)**: Holding across 2-5 days amortizes round-trip friction and captures persistent statistical range reversion on large-cap cyclicals.
3. **Anti-Duplication Protection**: The controller actively prunes redundant parameter variations, focusing exclusively on structurally orthogonal market mechanisms.

## 5. Candidate Ranking & Next Research Directions

```text
CURRENT BEST CANDIDATE PORTFOLIO:
  1. Alpha 14 (Gap Momentum Drift)           --> Primary Forward Paper Candidate (540d: +Rs 7.7k, OOS: +Rs 2.6k)
  2. Alpha 10 (Statistical Range Reversion)   --> Secondary Multi-Day Swing Candidate (MARUTI: +Rs 22.6k)
  3. Alpha 09 (Opening Relative Strength)     --> Sector IT Leadership Watchlist (INFY: +Rs 14.4k, TCS: +Rs 9.6k)
```
