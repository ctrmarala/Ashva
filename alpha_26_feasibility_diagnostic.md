# Alpha 26 Feasibility Diagnostic Report
**Hypothesis: Intraday Relative Strength Persistence vs Cross-Sectional Benchmark**

- **Date**: `2026-08-18`
- **Scope**: Step 0 Pre-Implementation Feasibility Diagnostic
- **Dataset**: `NIFTY-14` Universe across `180 Days` (15-Minute Bars)
- **Sample Size**: `21,442` Intraday Decision Bars (09:45 to 14:00 IST)
- **Status**: 🔴 **REJECTED AT STEP 0 (Hypothesis Does Not Warrant Implementation)**

---

## 1. Experimental Setup & Definitions

We tested whether continuous outperformance/underperformance versus the contemporaneous universe median return across $K=3$ consecutive 15-minute bars (45 minutes of sustained relative movement) exhibits statistically and economically meaningful forward continuation:

1. **Relative Return**: $\alpha_{s, t} = R_{s, t} - R_{\text{universe\_median}, t}$
2. **Persistent Leader**: $\alpha_{s, t} > 0$ for 3 consecutive bars within the same session.
3. **Transient Leader**: $\alpha_{s, t} > 0$ on bar $t$, but not on bar $t-1$.
4. **Persistent Laggard**: $\alpha_{s, t} < 0$ for 3 consecutive bars within the same session.
5. **Transient Laggard**: $\alpha_{s, t} < 0$ on bar $t$, but not on bar $t-1$.
6. **Forward Horizon Metrics**:
   - $F_1$: Forward 1-bar return (Next Open to Next Close)
   - $F_4$: Forward 4-bar return (Next Open to Bar $t+4$ Close, 1-Hour Horizon)
   - $F_{\text{EOD}}$: Forward holding to 15:15 IST Square-Off

---

## 2. Diagnostic Empirical Results

### A. Baseline Persistence vs Transient Shocks (21,442 Observations)

| Category | Observation Count | Forward 1-Bar Mean | Forward 1-Bar Win Rate | Forward 4-Bar Mean | Forward 4-Bar Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistent Leader (3+ bars)** | **3,575** | +0.05 bps | 48.0% | **+0.56 bps** | **47.2%** |
| **Transient Leader (1 bar)** | 7,186 | -0.13 bps | 47.4% | +0.52 bps | 47.4% |
| **Persistent Laggard (3+ bars)** | **3,493** | +0.70 bps | 50.5% | **+2.26 bps** | **53.3%** |
| **Transient Laggard (1 bar)** | 7,188 | -0.11 bps | 49.8% | -0.19 bps | 51.5% |

### B. Magnitude-Conditioned Persistence (Cumulative $\sum \alpha_{s, t} \ge \pm 0.20\%$ to $\pm 0.50\%$)

| Magnitude Tier | Event Count | Forward 4-Bar Mean (bps) | 4-Bar Win Rate | Forward EOD Mean (bps) | EOD Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strong Leader ($\sum \alpha \ge +0.50\%$)** | 2,139 | **+0.84 bps** | 48.3% | **+0.35 bps** | 47.1% |
| **Strong Laggard ($\sum \alpha \le -0.50\%$)** | 1,770 | **-0.24 bps** | 51.1% | **-5.43 bps** | 48.8% |
| **Moderate Leader ($\sum \alpha \ge +0.20\%$)** | 4,793 | **+1.41 bps** | 47.7% | **+1.90 bps** | 48.3% |
| **Moderate Laggard ($\sum \alpha \le -0.20\%$)** | 5,028 | **+1.27 bps** | 52.4% | **+0.19 bps** | 52.6% |

---

## 3. Scientific Assessment & Friction Math

1. **The Friction Hurdle**:
   - Under standard Indian equity intraday transaction friction (STT 0.025% on sell, GST 18%, Stamp Duty 0.003%, SEBI & NSE turnover charges) plus realistic execution slippage of 3.0 bps, total round-trip friction is **6.5 to 7.5 basis points**.
2. **Gross Return vs Transaction Friction**:
   - The maximum gross forward edge observed for persistent leaders is **+0.84 to +1.41 bps**.
   - Net expected payoff after friction is **systematically negative (-5.1 to -6.1 bps per trade)**.
3. **Win Rate Deficit**:
   - Forward win rates on persistent leaders are **$47.1\%$ to $48.3\%$** (persistently below 50%), confirming that generic midday intraday momentum in Indian mega-caps experiences mean-reverting resistance.
4. **Why Alpha 09 Succeeded While Alpha 26 Fails**:
   - **Alpha 09 (Opening Relative Strength)** isolates the **09:15-09:30 opening auction window**, where institutional order flow establishes multi-hour directional trends with high volume expansion.
   - **Alpha 26 (General Intraday Relative Strength)** attempts to trade midday drift (10:00 to 14:00), which lacks the institutional volume catalyst and gets chopped up by liquidity replenishment.

---

## 4. Final Recommendation

# 🛑 REJECT AT STEP 0 (DO NOT IMPLEMENT)

In accordance with our multiple-testing discipline, because the preliminary market data diagnostic reveals an edge of $< 1.5\text{ bps}$ against a $7.0\text{ bps}$ cost hurdle, **we reject Alpha 26 before code implementation**.

This prevents data-mining, saves compute, and preserves the scientific integrity of the research pipeline.
