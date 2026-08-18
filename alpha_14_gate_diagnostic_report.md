# ASHVA ALPHA 14 REGIME GATE DIAGNOSTIC REPORT

> **Strategy**: `ALPHA_14_GAP_MOMENTUM_DRIFT` (Frozen Execution & Sizing Parameters)
> **Universe**: `NIFTY14` (14 Assets) | **Lookback**: `540 Days`
> **Cost Engine**: Indian Statutory Taxes + 3.0 bps Slippage

## 1. Comparative Gate Performance Matrix (Baseline vs Gate A vs Gate B vs Gate A+B)

| Configuration                 |   Trades_540d |   Retained_Pct |   Net_PnL_540d_INR |   Basket_ROI_Pct |   Net_PF |   Win_Rate_Pct |   Sharpe |   Max_DD_Pct |   W1_Favorable_PnL |   W1_PF |   W2_PnL |   W3_W4_Choppy_PnL |   W3_W4_Trades |   W5_OOS_PnL |   W5_PF |
|:------------------------------|--------------:|---------------:|-------------------:|-----------------:|---------:|---------------:|---------:|-------------:|-------------------:|--------:|---------:|-------------------:|---------------:|-------------:|--------:|
| 1. Baseline (No Gates)        |           147 |          100   |          -10805.3  |            -0.15 |     1.51 |           39.5 |    -1.29 |         0.22 |            1088.78 |    2.37 |  -621.31 |          -10518.2  |             55 |      -754.61 |    1.85 |
| 2. Gate A Only (Macro Regime) |            82 |           55.8 |           -8091.44 |            -0.12 |     1.3  |           40.2 |    -1.04 |         0.17 |            -279.37 |    1.71 |  -943.38 |           -7986.66 |             33 |      1117.97 |    2.69 |
| 3. Gate B Only (Asset Trend)  |            80 |           54.4 |           -7626.49 |            -0.11 |     1.33 |           43.8 |    -1.19 |         0.15 |             529.16 |    1.97 | -1302.15 |           -8341.66 |             30 |      1488.15 |    2.49 |
| 4. Gate A + Gate B (Combined) |            64 |           43.5 |           -7225.76 |            -0.1  |     1.23 |           42.2 |    -1.17 |         0.14 |             -34.4  |    1.75 |  -548.82 |           -7659.81 |             23 |      1017.27 |    2.61 |

## 2. Key Diagnostic Findings

- **Gate A (Macro Regime Gate)**: Requires prior-day Nifty Market Breadth (% of stocks > 20d SMA) >= 50% for Longs and < 50% for Shorts.
- **Gate B (Asset Multi-Day Trend Gate)**: Requires the target stock's prior-day close to be > 20d SMA for Longs and < 20d SMA for Shorts.
- **Gate A+B (Combined Gate)**: Both macro and micro trend alignment required at 09:30 AM.

