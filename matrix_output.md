# ASHVA ALPHA RESEARCH MATRIX AUDIT — [RESEARCH MODE]

> **[*] RESEARCH RUN -- INSTITUTIONAL AUDIT WITH TEMPORAL OOS**

- **Execution Mode**: `RESEARCH`
- **Universe**: `NIFTY14` (14 Assets)
- **Audited Strategies**: `1 Alphas` (alpha_24)
- **Historical Lookback**: `540 Days` (IS: `420d` | Untouched OOS: `120d`)
- **Capital Deployment**: `Rs 500,000/Asset` (Total Basket Capital = `Rs 7,000,000`)
- **Cost Model**: Indian Statutory Taxes (STT, Exchange, GST, SEBI, Stamp Duty) + 3.0 bps Slippage
- **Timestamp**: `2026-08-18 23:16:44 IST`

## 1. Strategy Summary Matrix (1 Alphas)

| Alpha_ID   | Strategy                           |   Net_PnL_INR |   Basket_ROI_Pct |   Ann_Return_Pct |   Sharpe |   Max_DD_Pct |   PF_Lookback |   PF_60d |   Trades |   Win_Rate_Pct |   Avg_Trade_INR |   Recency_Q | Observed_Positive_Cluster                                                      |   Inter_Alpha_Corr |   Trade_PnL_Corr | Strategy_Classification           |
|:-----------|:-----------------------------------|--------------:|-----------------:|-----------------:|---------:|-------------:|--------------:|---------:|---------:|---------------:|----------------:|------------:|:-------------------------------------------------------------------------------|-------------------:|-----------------:|:----------------------------------|
| alpha_24   | ALPHA_24_VOLATILITY_VACUUM_RELEASE |       -154659 |            -2.21 |            -1.49 |    -2.48 |         2.62 |          0.66 |     0.97 |     1281 |             34 |          -120.7 |       -0.37 | 🟢 3/14 Assets: TATASTEEL (+Rs 6,161), MARUTI (+Rs 3,450), HDFCBANK (+Rs 1,935) |                  0 |                0 | 🔍 Candidate Asset Edges (3 Pairs) |

## 2. Candidate Alpha × Asset Edges (Observed Positive Pairs)

| Alpha_ID   | Strategy                           | Symbol    |   Net_PnL_INR |   Net_ROI_Pct |   Trades |   Win_Rate_Pct |   PF_Lookback |   PF_60d |   Sharpe |   Max_DD_Pct |   Recency_Q | Status     |
|:-----------|:-----------------------------------|:----------|--------------:|--------------:|---------:|---------------:|--------------:|---------:|---------:|-------------:|------------:|:-----------|
| alpha_24   | ALPHA_24_VOLATILITY_VACUUM_RELEASE | TATASTEEL |       6160.68 |          1.23 |      104 |           43.3 |          1.21 |     1.89 |     0.34 |         2.66 |        0.05 | 🟢 Positive |
| alpha_24   | ALPHA_24_VOLATILITY_VACUUM_RELEASE | MARUTI    |       3449.82 |          0.69 |      105 |           35.2 |          1.22 |     2.18 |     0.25 |         1.53 |        0.14 | 🟢 Positive |
| alpha_24   | ALPHA_24_VOLATILITY_VACUUM_RELEASE | HDFCBANK  |       1934.92 |          0.39 |       69 |           37.7 |          1.17 |     2.17 |     0.21 |         1.58 |        0.47 | 🟢 Positive |

## 3. Temporal Out-Of-Sample (OOS) Validation (120 Days Untouched Test Period)

| Alpha_ID   | Strategy                           |   IS_Lookback_Days |   IS_Trades |   IS_Net_PnL_INR |   IS_Basket_ROI_Pct |   OOS_Untouched_Days |   OOS_Trades |   OOS_Win_Rate_Pct |   OOS_Net_PnL_INR |   OOS_Basket_ROI_Pct |   OOS_Sharpe | OOS_Status     |
|:-----------|:-----------------------------------|-------------------:|------------:|-----------------:|--------------------:|---------------------:|-------------:|-------------------:|------------------:|---------------------:|-------------:|:---------------|
| alpha_24   | ALPHA_24_VOLATILITY_VACUUM_RELEASE |                420 |         948 |         -98735.9 |               -1.41 |                  120 |          333 |               34.2 |          -54967.1 |                -0.79 |        -3.51 | 🔴 Negative OOS |

## 4. Full 2D Alpha × Asset Interaction Grid (1 Alphas × 14 Assets)

|                                    | INFY                     | TCS                       | ICICIBANK               | HDFCBANK                | SBIN                      | AXISBANK                  | KOTAKBANK                | RELIANCE                 | LT                       | TATASTEEL                | BHARTIARTL               | BAJFINANCE               | MARUTI                   | SUNPHARMA                |
|:-----------------------------------|:-------------------------|:--------------------------|:------------------------|:------------------------|:--------------------------|:--------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|
| ALPHA_24_VOLATILITY_VACUUM_RELEASE | 🔴 -Rs 18.2k (69T|PF:0.5) | 🔴 -Rs 20.3k (115T|PF:0.6) | 🔴 -Rs 0.6k (71T|PF:1.1) | 🟢 +Rs 1.9k (69T|PF:1.2) | 🔴 -Rs 15.1k (108T|PF:0.7) | 🔴 -Rs 15.1k (103T|PF:0.7) | 🔴 -Rs 24.4k (67T|PF:0.4) | 🔴 -Rs 5.8k (115T|PF:0.9) | 🔴 -Rs 15.1k (93T|PF:0.6) | 🟢 +Rs 6.2k (104T|PF:1.2) | 🔴 -Rs 25.8k (99T|PF:0.4) | 🔴 -Rs 13.0k (86T|PF:0.8) | 🟢 +Rs 3.4k (105T|PF:1.2) | 🔴 -Rs 12.7k (77T|PF:0.6) |

