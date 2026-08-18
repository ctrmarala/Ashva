# Ashva Quantitative Platform — Existing Alpha Baseline & Discovery Map
**Authoritative Post-Lifecycle Research Registry & Mechanism Landscape**

- **Audit Lookback**: `540 Days` (IS: `420d` | Untouched OOS: `120d`)
- **Universe**: `NIFTY14` Liquid Mega-Caps (Basket Capital: `₹7,000,000`)
- **Cost Model**: Full Indian Statutory Schedule (STT, GST, Stamp Duty, SEBI, NSE fees, ₹20 cap) + 3.0 bps Slippage
- **Factory Status**: 🔒 **FROZEN** (All 23 strategies verified persistent under `BacktestEngine`)

---

## 1. Executive Summary & Qualification Framework

Following the correction of the 1-bar position lifecycle defect, all 23 strategies have been re-baselined under their true economic holding periods.

### The Three Tiers of Validation

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. ENGINEERING VALIDATED (23 / 23 PASSED)                                   │
│    All 23 strategies correctly implement the BaseHypothesis contract,       │
│    maintain intraday position state, and pass automated CI contract tests.  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. RESEARCH VALIDATED (4 Candidates Identified)                             │
│    Strategies demonstrating positive OOS PnL, persistent asset clusters, or │
│    falsifiable economic mechanisms after Indian statutory friction.         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. PRODUCTION VALIDATED (0 Alphas Promoted)                                 │
│    Zero alphas are deployed with live capital. All candidates must complete │
│    forward paper trading and portfolio diversification qualification.       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Master 23-Alpha Research Status Table

| Alpha ID | Strategy Name | 540d Net PnL | 540d Sharpe | 120d OOS Trades | 120d OOS PnL | 120d OOS Sharpe | Positive Assets | Research Status | Classification Reason |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **alpha_14** | `ALPHA_14_GAP_MOMENTUM_DRIFT` | **+₹7,743** | **+0.21** | **131** | **+₹2,603** | **+0.26** | 7 / 14 | 📝 **Paper Candidate** | 540d profitable (+₹7.7k), positive OOS (+₹2.6k across 131T), 4/5 windows profitable, strong RELIANCE edge (+₹18.5k, PF 3.70). |
| **alpha_10** | `ALPHA_10_STATISTICAL_RANGE_REVERSION` | -₹51,181 | -0.78 | **51** | **+₹2,581** | **+0.12** | 3 / 14 | 🟡 **Research Candidate** | Multi-day swing persistence turned 120d OOS positive (+₹2.58k, 51T); exceptional MARUTI edge (+₹22.6k, 100% WR). |
| **alpha_04** | `ALPHA_04_GAP_AND_GO` | **+₹5,261** | **+0.70** | **4** | **+₹1,081** | **+0.76** | 5 / 14 | 🔬 **Discovery (Low Freq)** | 540d profitable (+₹5.2k, Sharpe 0.70, PF 99.0); small sample ($N=7$ total, 4 in OOS) prevents promotion to paper trading. |
| **alpha_09** | `ALPHA_09_OPENING_RELATIVE_STRENGTH` | -₹11,402 | -0.23 | **111** | -₹2,262 | -0.20 | 8 / 14 | 🟡 **Research Candidate** | Broadest positive cluster (8/14 assets profitable); high edge in IT (INFY +₹14.4k, TCS +₹9.6k) and ICICIBANK (+₹6.2k). |
| **alpha_05** | `ALPHA_05_OPENING_DRIVE_PULLBACK` | -₹21,970 | -0.98 | 35 | -₹56 | -0.01 | 3 / 14 | 🔬 **Discovery** | Near-breakeven OOS (-₹56); positive edges in RELIANCE (+₹2.1k) and LT (+₹1.4k); insufficient aggregate strength. |
| **alpha_03** | `ALPHA_03_VWAP_REVERSION` | -₹34,120 | -2.29 | 45 | -₹5,803 | -1.86 | 3 / 14 | 🔬 **Discovery** | Modest asset edges on TCS (+₹3.1k), ICICIBANK (+₹1.7k), AXISBANK (+₹1.5k); negative overall drag. |
| **alpha_16** | `ALPHA_16_INSIDE_DAY_BREAKOUT` | -₹50,093 | -1.14 | 87 | -₹15,383 | -1.81 | 5 / 14 | 🔬 **Discovery** | Positive edges in TATASTEEL (+₹10.2k, PF 1.96) and BHARTIARTL (+₹4.0k); negative basket drift. |
| **alpha_11** | `ALPHA_11_DONCHIAN_BREAKOUT` | -₹53,627 | -1.16 | 8 | -₹9,115 | -1.38 | 4 / 14 | 🔬 **Discovery** | Multi-day breakout works on SBIN (+₹6.3k, PF 1.44) and RELIANCE (+₹2.3k); fails on high-beta names. |
| **alpha_08** | `ALPHA_08_OPENING_IMBALANCE` | -₹82,249 | -1.69 | 152 | -₹19,962 | -1.52 | 3 / 14 | 🔬 **Discovery** | Positive on INFY (+₹9.3k) and AXISBANK (+₹4.8k); high churn on remaining universe. |
| **alpha_12** | `ALPHA_12_EUROPEAN_OPEN_MOMENTUM` | -₹117,333 | -1.45 | 212 | -₹12,602 | -0.99 | 5 / 14 | 🔬 **Discovery** | Midday European open momentum shows edges on SUNPHARMA (+₹9.7k) and ICICIBANK (+₹7.9k); drag from choppy sessions. |
| **alpha_15** | `ALPHA_15_NR7_VOLATILITY_EXPANSION` | -₹115,633 | -3.59 | 152 | -₹41,799 | -5.31 | 1 / 14 | 🔴 **Low-Priority Archive** | Severe OOS decay (-₹41.8k); only BAJFINANCE positive (+₹7.7k); false breakout rate too high after costs. |
| **alpha_13** | `ALPHA_13_HTF_ALIGNED_ORB` | -₹157,126 | -2.82 | 231 | -₹55,015 | -4.95 | 0 / 14 | 🔴 **Low-Priority Archive** | Zero profitable assets; HTF EMA20 alignment fails to filter chop. |
| **alpha_18** | `ALPHA_18_THREE_DAY_TREND_ORB` | -₹201,902 | -3.59 | 252 | -₹49,370 | -4.62 | 1 / 14 | 🔴 **Low-Priority Archive** | Continuous degradation (-₹201.9k 540d PnL); 3-day trend context provides no statistical edge. |
| **alpha_17** | `ALPHA_17_VOLUME_SHOCK_MOMENTUM` | -₹207,849 | -3.55 | 392 | -₹62,198 | -5.10 | 1 / 14 | 🔴 **Low-Priority Archive** | High trade frequency ($N=1,406$) incurs massive tax and slippage drag; only INFY positive (+₹6.8k). |
| **alpha_02** | `ALPHA_02_AUCTION_ORB` | -₹258,818 | -2.38 | 561 | -₹92,947 | -3.97 | 3 / 14 | 🔴 **Low-Priority Archive** | High churn ($N=2,246$); positive in TCS (+₹10.3k), RELIANCE (+₹2.0k), BHARTIARTL (+₹2.4k), but aggregate loss is steep. |
| **alpha_22** | `ALPHA_22_APEX_MOMENTUM` | -₹298,053 | -4.68 | 423 | -₹97,704 | -7.20 | 0 / 14 | 🔴 **Low-Priority Archive** | Zero profitable assets; multi-filter ORB over-fits in-sample and collapses out-of-sample. |
| **alpha_07** | `ALPHA_07_OPENING_VOLATILITY_EXPANSION` | -₹301,845 | -3.62 | 348 | -₹117,916 | -7.57 | 1 / 14 | 🔴 **Low-Priority Archive** | Severe decay (-₹301.8k); only HDFCBANK positive (+₹9.1k). |
| **alpha_21** | `ALPHA_21_HIGH_VELOCITY_MOMENTUM` | -₹347,320 | -5.66 | 384 | -₹93,632 | -7.61 | 0 / 14 | 🔴 **Low-Priority Archive** | Zero profitable assets; velocity thresholds fail to account for midday mean reversion. |
| **alpha_20** | `ALPHA_20_VWAP_TREND_CONTINUATION` | -₹502,510 | -6.48 | 541 | -₹134,896 | -7.35 | 0 / 14 | 🔴 **Low-Priority Archive** | Severe churn ($N=2,494$); VWAP crosses generate whipsaws in non-trending Indian large caps. |
| **alpha_01** | `ALPHA_01_TRENDSURFER` | -₹642,338 | -4.95 | 878 | -₹144,613 | -5.50 | 0 / 14 | 🔴 **Low-Priority Archive** | Classical Supertrend/EMA trend surfing is unviable on 15m Indian equities after statutory friction. |
| **alpha_23** | `ALPHA_23_VELOCITY_50_SCANNER` | -₹799,822 | -8.10 | 1061 | -₹219,637 | -9.67 | 0 / 14 | 🔴 **Low-Priority Archive** | Massive trade volume ($N=4,613$) generates ₹3.8L in transaction friction. |
| **alpha_06** | `ALPHA_06_PDH_PDL_SWEEP` | -₹2,176,231 | -14.91 | 3454 | -₹591,121 | -15.20 | 0 / 14 | 🔴 **Low-Priority Archive** | Extreme trade churn ($N=14,497$); structural failure under realistic transaction costs. |
| **alpha_19** | `ALPHA_19_POWER_HOUR_MOMENTUM` | -₹6,524 | -1.39 | 2 | -₹1,349 | -2.68 | 1 / 14 | 🔴 **Low-Priority Archive** | Very low trade frequency ($N=11$); late-day 14:00 breakout has negative edge across all assets except BHARTIARTL. |

---

## 3. Strongest Alpha × Asset Empirical Observations

These observed positive pairs represent the foundational empirical dataset for future candidate tuning and alpha-specific universe selection:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TOP 10 ALPHA × ASSET EDGES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. MARUTI     × Alpha 10 (Stat Range Reversion)   : +₹22,631 | PF 99.0 | 7T │
│ 2. RELIANCE   × Alpha 14 (Gap Momentum Drift)     : +₹18,512 | PF 3.70 | 24T│
│ 3. INFY       × Alpha 09 (Opening Relative Str)   : +₹14,403 | PF 1.89 | 41T│
│ 4. TCS        × Alpha 02 (Auction ORB)            : +₹10,334 | PF 1.18 |171T│
│ 5. TATASTEEL  × Alpha 16 (Inside Day Breakout)    : +₹10,230 | PF 1.96 | 35T│
│ 6. SUNPHARMA  × Alpha 12 (European Open Momentum) : +₹9,730  | PF 1.56 | 63T│
│ 7. TCS        × Alpha 09 (Opening Relative Str)   : +₹9,591  | PF 1.89 | 33T│
│ 8. INFY       × Alpha 08 (Opening Imbalance)      : +₹9,281  | PF 1.41 | 54T│
│ 9. HDFCBANK   × Alpha 07 (Opening Vol Expansion)  : +₹9,142  | PF 1.29 |105T│
│ 10. BAJFINANCE× Alpha 15 (NR7 Volatility Expansion): +₹7,710 | PF 2.99 | 20T│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Known Market Mechanisms & Regime Clues

### 1. Gap Momentum Drift (Alpha 14)
- **Economic Inefficiency**: Institutional pre-open order flow imbalances create opening gaps that experience follow-through drift between 09:30 and 15:15.
- **Regime Sensitivity**:
  - **Favorable**: Expansion and trending sessions with strong directional conviction ($p = 0.0028$ on body/range ratio).
  - **Unfavorable**: Choppy, mean-reverting sessions with rejection wicks ($p = 0.0101$).
- **Key Insight**: Holding until 15:15 EOD square-off (rather than 1-bar scalping) is the foundational driver of its ₹7.7k 540d / ₹2.6k OOS edge.

### 2. Multi-Day Statistical Range Reversion (Alpha 10)
- **Economic Inefficiency**: Mega-cap equities (e.g. MARUTI) over-extend beyond multi-day statistical standard deviation bands and revert to mean over 2 to 5 days.
- **Key Insight**: Swing holding (without forced 15:15 EOD square-off) turned OOS positive (+₹2,581).

### 3. Opening Relative Strength Dispersion (Alpha 09)
- **Economic Inefficiency**: Stocks showing strong opening relative strength vs NIFTY-50 index continue outperforming during intraday hours.
- **Key Insight**: Works across 8 of 14 stocks, especially high-beta IT (INFY, TCS) and Private Banks (ICICIBANK, AXISBANK).

### 4. Extreme Gap and Go (Alpha 04)
- **Economic Inefficiency**: Gaps $>0.75\%$ with opening expansion have a 71.4% win rate and 0.70 Sharpe.
- **Key Insight**: High precision, but too rare ($N=7$ trades in 540 days) for standalone production.

---

## 5. Mechanism Territory Map: Explored vs White-Space

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 🔴 HEAVILY EXPLORED TERRITORY (Do NOT Generate Variations)                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ • Classical ORB Breakouts (Alphas 02, 13, 18, 21, 22, 23)                    │
│ • Previous Day High/Low Sweeps (Alpha 06)                                    │
│ • Single-Bar VWAP Crosses (Alpha 20)                                         │
│ • Naive Supertrend / Moving Average Surfing (Alpha 01)                       │
│ • High-Frequency Volume Shock Momentum without Regime Gate (Alpha 17)        │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 🟢 WHITE-SPACE TERRITORY (Target Areas for Alpha 24+)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│ 1. Cross-Sectional Sector Dispersion & Leader/Laggard Momentum               │
│    (Exploits sector-rotation momentum e.g. Nifty IT vs Nifty Bank)           │
│                                                                              │
│ 2. Order Book & Auction Imbalance Exhaustion / Fade                          │
│    (Fading false opening expansions that exhibit rejection wicks)            │
│                                                                              │
│ 3. Volatility Squeeze & Compression Expansion                                │
│    (Multi-timeframe Bollinger/Keltner squeeze with directional breakout)     │
│                                                                              │
│ 4. Intraday Statistical Cointegration & Pairs Drift                          │
│    (Trading structural spread dislocations between highly correlated pairs)  │
│                                                                              │
│ 5. Time-Weighted Volume Profile Value Area Reversals (VAH/VAL)               │
│    (Mean reversion from Value Area boundaries back to POC)                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Actionable Discovery Next Steps

1. **Retain Active Research Candidates**:
   - `Alpha 14` (Paper Candidate: Gap Momentum Drift)
   - `Alpha 10` (Research Candidate: Swing Range Reversion)
   - `Alpha 09` (Research Candidate: Opening Relative Strength)
   - `Alpha 04` (Discovery: Low-Frequency Gap and Go)
2. **Freeze Archive Pool (19 Alphas)**:
   - Preserved in repository for mechanism reference, but excluded from daily development loops to preserve compute.
3. **Launch Alpha Discovery Pipeline**:
   - Begin authoring `Alpha 24` targeting **White-Space Territory** (e.g. Cross-Sectional Sector Momentum or Squeeze Expansion) using the frozen factory.
