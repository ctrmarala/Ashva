# ASHVA ALPHA RESEARCH MATRIX AUDIT — [RESEARCH MODE]

> **[*] RESEARCH RUN -- INSTITUTIONAL AUDIT WITH TEMPORAL OOS**

- **Execution Mode**: `RESEARCH`
- **Universe**: `NIFTY14` (14 Assets)
- **Audited Strategies**: `1 Alphas` (alpha_25)
- **Historical Lookback**: `540 Days` (IS: `420d` | Untouched OOS: `120d`)
- **Capital Deployment**: `Rs 500,000/Asset` (Total Basket Capital = `Rs 7,000,000`)
- **Cost Model**: Indian Statutory Taxes (STT, Exchange, GST, SEBI, Stamp Duty) + 3.0 bps Slippage
- **Timestamp**: `2026-08-18 23:22:58 IST`

## 1. Strategy Summary Matrix (1 Alphas)

| Alpha_ID   | Strategy                                    |   Net_PnL_INR |   Basket_ROI_Pct |   Ann_Return_Pct |   Sharpe |   Max_DD_Pct |   PF_Lookback |   PF_60d |   Trades |   Win_Rate_Pct |   Avg_Trade_INR |   Recency_Q | Observed_Positive_Cluster     |   Inter_Alpha_Corr |   Trade_PnL_Corr | Strategy_Classification           |
|:-----------|:--------------------------------------------|--------------:|-----------------:|-----------------:|---------:|-------------:|--------------:|---------:|---------:|---------------:|----------------:|------------:|:------------------------------|-------------------:|-----------------:|:----------------------------------|
| alpha_25   | ALPHA_25_CROSS_SECTIONAL_RESIDUAL_REVERSION |       -133950 |            -1.91 |            -1.29 |    -3.73 |          1.9 |          0.46 |     0.14 |      609 |           28.7 |            -220 |       -0.22 | 🟢 1/14 Assets: LT (+Rs 1,002) |                  0 |                0 | 🔍 Candidate Asset Edges (1 Pairs) |

## 2. Candidate Alpha × Asset Edges (Observed Positive Pairs)

| Alpha_ID   | Strategy                                    | Symbol   |   Net_PnL_INR |   Net_ROI_Pct |   Trades |   Win_Rate_Pct |   PF_Lookback |   PF_60d |   Sharpe |   Max_DD_Pct |   Recency_Q | Status     |
|:-----------|:--------------------------------------------|:---------|--------------:|--------------:|---------:|---------------:|--------------:|---------:|---------:|-------------:|------------:|:-----------|
| alpha_25   | ALPHA_25_CROSS_SECTIONAL_RESIDUAL_REVERSION | LT       |        1002.5 |           0.2 |       28 |           57.1 |          1.44 |        0 |     0.25 |         0.43 |        0.04 | 🟢 Positive |

## 3. Temporal Out-Of-Sample (OOS) Validation (120 Days Untouched Test Period)

| Alpha_ID   | Strategy                                    |   IS_Lookback_Days |   IS_Trades |   IS_Net_PnL_INR |   IS_Basket_ROI_Pct |   OOS_Untouched_Days |   OOS_Trades |   OOS_Win_Rate_Pct |   OOS_Net_PnL_INR |   OOS_Basket_ROI_Pct |   OOS_Sharpe | OOS_Status     |
|:-----------|:--------------------------------------------|-------------------:|------------:|-----------------:|--------------------:|---------------------:|-------------:|-------------------:|------------------:|---------------------:|-------------:|:---------------|
| alpha_25   | ALPHA_25_CROSS_SECTIONAL_RESIDUAL_REVERSION |                420 |         495 |          -120639 |               -1.72 |                  120 |          114 |                 36 |          -13536.6 |                -0.19 |        -2.16 | 🔴 Negative OOS |

## 4. Full 2D Alpha × Asset Interaction Grid (1 Alphas × 14 Assets)

|                                             | INFY                    | TCS                      | ICICIBANK               | HDFCBANK                | SBIN                     | AXISBANK                 | KOTAKBANK               | RELIANCE                 | LT                      | TATASTEEL                | BHARTIARTL              | BAJFINANCE              | MARUTI                  | SUNPHARMA                |
|:--------------------------------------------|:------------------------|:-------------------------|:------------------------|:------------------------|:-------------------------|:-------------------------|:------------------------|:-------------------------|:------------------------|:-------------------------|:------------------------|:------------------------|:------------------------|:-------------------------|
| ALPHA_25_CROSS_SECTIONAL_RESIDUAL_REVERSION | 🔴 -Rs 2.1k (18T|PF:0.7) | 🔴 -Rs 10.9k (50T|PF:0.4) | 🔴 -Rs 1.5k (15T|PF:0.7) | 🔴 -Rs 3.8k (35T|PF:0.6) | 🔴 -Rs 30.1k (73T|PF:0.2) | 🔴 -Rs 15.2k (67T|PF:0.3) | 🔴 -Rs 9.9k (38T|PF:0.2) | 🔴 -Rs 10.1k (34T|PF:0.3) | 🟢 +Rs 1.0k (28T|PF:1.4) | 🔴 -Rs 26.7k (66T|PF:0.2) | 🔴 -Rs 3.0k (45T|PF:0.8) | 🔴 -Rs 1.8k (36T|PF:1.0) | 🔴 -Rs 8.4k (53T|PF:0.4) | 🔴 -Rs 11.6k (51T|PF:0.5) |

