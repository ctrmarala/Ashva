# ASHVA ALPHA 14 SIGNAL DECOMPOSITION & ATTRIBUTION REPORT

> **Strategy**: `ALPHA_14_GAP_MOMENTUM_DRIFT` (Frozen Parameters)
> **Total Trades Analyzed**: `370 Trades` across `14 Stocks` over `540 Days`
> **Cost Engine**: Indian Statutory Taxes + 3.0 bps Slippage

## 1. Feature Comparison: Winning Trades vs Losing Trades

| Feature                     |   Winners_Mean |   Winners_Median |   Losers_Mean |   Losers_Median |   Difference |   T_Statistic |   P_Value | Significance   |
|:----------------------------|---------------:|-----------------:|--------------:|----------------:|-------------:|--------------:|----------:|:---------------|
| Overnight Gap (%)           |           1.24 |             0.81 |          1.62 |            0.98 |        -0.38 |         -2.75 |    0.0064 | ⭐⭐ (p < 0.01)  |
| Gap / Daily ATR Ratio       |           0.57 |             0.41 |          0.73 |            0.47 |        -0.16 |         -2.64 |    0.0087 | ⭐⭐ (p < 0.01)  |
| Bar 1 Body/Range (%)        |          61.94 |            66.69 |         53.51 |           56.18 |         8.43 |          3.01 |    0.0028 | ⭐⭐ (p < 0.01)  |
| 09:15 RVOL (x TOD)          |       26028.6  |             1.37 |       1002.18 |            1    |     25026.4  |          1.05 |    0.2961 | ns (p >= 0.05) |
| Normalized ATR (%)          |           2.15 |             2.16 |          2.11 |            2.16 |         0.03 |          0.71 |    0.479  | ns (p >= 0.05) |
| Adverse Rejection Wick (%)  |          29.56 |            21.69 |         37.49 |           31.43 |        -7.93 |         -2.59 |    0.0101 | ⭐ (p < 0.05)   |
| Gap Held Entire Bar 1 (%)   |           0.91 |             1    |          0.89 |            1    |         0.03 |          0.81 |    0.4206 | ns (p >= 0.05) |
| Trade Duration (Bars)       |          12.35 |            12    |          6.51 |            1    |         5.85 |          5.26 |    0      | ⭐⭐ (p < 0.01)  |
| Max Favorable Excursion (%) |           1.58 |             1.28 |          0.3  |            0.18 |         1.28 |         10.43 |    0      | ⭐⭐ (p < 0.01)  |
| Max Adverse Excursion (%)   |          -0.3  |            -0.17 |         -0.71 |           -0.38 |         0.4  |          6.5  |    0      | ⭐⭐ (p < 0.01)  |

## 2. Regime-by-Regime Signal Microstructure (Favorable W1 vs Choppy W3/W4 vs OOS W5)

| Window   |   Trades |   Win_Rate_Pct |   Net_PnL_INR |   Avg_Gap_Pct |   Avg_Gap_To_ATR |   Avg_Body_Ratio_Pct |   Avg_RVOL |   Avg_Norm_ATR |   Gap_Held_Pct |   SL_Exits_Pct |   TP_Exits_Pct |   EOD_Exits_Pct |
|:---------|---------:|---------------:|--------------:|--------------:|-----------------:|---------------------:|-----------:|---------------:|---------------:|---------------:|---------------:|----------------:|
| W1       |       55 |           30.9 |       5580.64 |          0.79 |             0.36 |                 60.9 |   63611.6  |           2.24 |           96.4 |           54.5 |           16.4 |               0 |
| W2       |       84 |           28.6 |      -6744.56 |          1.1  |             0.62 |                 53.6 |       1.36 |           1.75 |           89.3 |           59.5 |           28.6 |               0 |
| W3       |       64 |           39.1 |        654.13 |          0.98 |             0.53 |                 55.8 |       1.71 |           1.77 |           79.7 |           42.2 |           26.6 |               0 |
| W4       |       58 |           41.4 |       4447.9  |          2.03 |             0.81 |                 56   |       1.76 |           2.51 |          100   |           24.1 |           50   |               0 |
| W5       |      109 |           32.1 |       3021.49 |          2.16 |             0.9  |                 56.7 |       1.37 |           2.36 |           86.2 |           44   |           43.1 |               0 |

## 3. Directional Asymmetry: Long vs Short Breakdown

| Side   |   Trades |   Trades_Pct |   Win_Rate_Pct |   Net_PnL_INR |   Net_PF |   Avg_Trade_INR |
|:-------|---------:|-------------:|---------------:|--------------:|---------:|----------------:|
| LONG   |      258 |         69.7 |           33.7 |       5039.31 |     1.05 |            19.5 |
| SHORT  |      112 |         30.3 |           33.9 |       1920.29 |     1.04 |            17.1 |

## 4. Execution Path & Exit Reason Distribution

| Exit_Reason   |   Trades |   Pct_Of_Total |   Net_PnL_INR |   Avg_Duration_Bars |   Avg_MFE_Pct |   Avg_MAE_Pct |
|:--------------|---------:|---------------:|--------------:|--------------------:|--------------:|--------------:|
| SIGNAL        |       75 |           20.3 |       6548.74 |                22.1 |          1.03 |         -0.82 |
| STOP_LOSS     |      169 |           45.7 |    -104101    |                 4.4 |          0.32 |         -0.66 |
| TAKE_PROFIT   |      126 |           34.1 |     104512    |                 5.9 |          1.1  |         -0.3  |

