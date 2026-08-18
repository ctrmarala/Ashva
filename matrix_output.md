# ASHVA ALPHA RESEARCH MATRIX AUDIT — [DEV MODE]

> **[!] DEVELOPMENT RUN -- NOT FOR RESEARCH CONCLUSIONS**

> [!WARNING]
> **DEVELOPMENT RUN ONLY**: This run used a truncated lookback and a 4-alpha subset for fast execution feedback. Do NOT use these metrics for formal research conclusions or live capital allocation.

- **Execution Mode**: `DEV`
- **Universe**: `NIFTY14` (14 Assets)
- **Audited Strategies**: `4 Alphas` (alpha_02, alpha_03, alpha_14, alpha_18)
- **Historical Lookback**: `120 Days` (IS: `90d` | Untouched OOS: `30d`)
- **Capital Deployment**: `Rs 500,000/Asset` (Total Basket Capital = `Rs 7,000,000`)
- **Cost Model**: Indian Statutory Taxes (STT, Exchange, GST, SEBI, Stamp Duty) + 3.0 bps Slippage
- **Timestamp**: `2026-08-18 23:01:16 IST`

## 1. Strategy Summary Matrix (4 Alphas)

| Alpha_ID   | Strategy                     |   Net_PnL_INR |   Basket_ROI_Pct |   Ann_Return_Pct |   Sharpe |   Max_DD_Pct |   PF_Lookback |   PF_60d |   Trades |   Win_Rate_Pct |   Avg_Trade_INR |   Recency_Q | Observed_Positive_Cluster                                                    |   Inter_Alpha_Corr |   Trade_PnL_Corr | Strategy_Classification           |
|:-----------|:-----------------------------|--------------:|-----------------:|-----------------:|---------:|-------------:|--------------:|---------:|---------:|---------------:|----------------:|------------:|:-----------------------------------------------------------------------------|-------------------:|-----------------:|:----------------------------------|
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  |      -1598.62 |            -0.02 |            -0.07 |    -0.15 |         0.26 |          1.07 |     0.94 |      122 |           32   |           -13.1 |       -0.02 | 🟢 7/14 Assets: INFY (+Rs 7,191), TCS (+Rs 2,811), BHARTIARTL (+Rs 2,411)     |               0.25 |             0.46 | 🔍 Candidate Asset Edges (7 Pairs) |
| alpha_03   | ALPHA_03_VWAP_REVERSION      |      -7173.12 |            -0.1  |            -0.31 |    -2.36 |         0.1  |          1.2  |     0.08 |       42 |           42.9 |          -170.8 |       -0.04 | 🟢 7/14 Assets: TATASTEEL (+Rs 781), BHARTIARTL (+Rs 399), AXISBANK (+Rs 307) |               0.05 |             0.09 | 🔍 Candidate Asset Edges (7 Pairs) |
| alpha_18   | ALPHA_18_THREE_DAY_TREND_ORB |     -48252.9  |            -0.69 |            -2.1  |    -4.6  |         0.71 |          0.44 |     0.06 |      239 |           22.2 |          -201.9 |       -0.38 | 🟢 1/14 Assets: KOTAKBANK (+Rs 168)                                           |               0.12 |             0.15 | 🔍 Candidate Asset Edges (1 Pairs) |
| alpha_02   | ALPHA_02_AUCTION_ORB         |    -100745    |            -1.44 |            -4.38 |    -4.22 |         1.49 |          0.72 |     0.83 |      572 |           40.4 |          -176.1 |       -0.25 | 🟢 5/14 Assets: TCS (+Rs 10,825), INFY (+Rs 8,726), RELIANCE (+Rs 5,862)      |               0.25 |             0.46 | 🔍 Candidate Asset Edges (5 Pairs) |

## 2. Candidate Alpha × Asset Edges (Observed Positive Pairs)

| Alpha_ID   | Strategy                     | Symbol     |   Net_PnL_INR |   Net_ROI_Pct |   Trades |   Win_Rate_Pct |   PF_Lookback |   PF_60d |   Sharpe |   Max_DD_Pct |   Recency_Q | Status     |
|:-----------|:-----------------------------|:-----------|--------------:|--------------:|---------:|---------------:|--------------:|---------:|---------:|-------------:|------------:|:-----------|
| alpha_02   | ALPHA_02_AUCTION_ORB         | TCS        |      10825.3  |          2.17 |       40 |           50   |          1.73 |     1.68 |     2.19 |         1.05 |        0.93 | 🟢 Positive |
| alpha_02   | ALPHA_02_AUCTION_ORB         | INFY       |       8725.91 |          1.75 |       38 |           57.9 |          1.45 |     1.74 |     1.61 |         1.52 |        0.92 | 🟢 Positive |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | INFY       |       7190.78 |          1.44 |        6 |           66.7 |          3.96 |    99    |     1.89 |         0.81 |        0.46 | 🟢 Positive |
| alpha_02   | ALPHA_02_AUCTION_ORB         | RELIANCE   |       5861.92 |          1.17 |       39 |           51.3 |          1.47 |     1.19 |     1.8  |         0.84 |        0.56 | 🟢 Positive |
| alpha_02   | ALPHA_02_AUCTION_ORB         | BHARTIARTL |       5473.14 |          1.09 |       35 |           54.3 |          1.37 |     0.77 |     1.31 |         1.42 |        0.26 | 🟢 Positive |
| alpha_02   | ALPHA_02_AUCTION_ORB         | BAJFINANCE |       3196.02 |          0.64 |       37 |           56.8 |          1.14 |     1.06 |     0.6  |         1.94 |        0.37 | 🟢 Positive |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | TCS        |       2810.55 |          0.56 |       13 |           23.1 |          0.77 |    99    |     1.12 |         0.69 |       -0.12 | 🟢 Positive |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | BHARTIARTL |       2411.44 |          0.48 |       12 |           41.7 |          1.62 |     1.4  |     2.02 |         0.27 |        0.21 | 🟢 Positive |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | HDFCBANK   |       1534.92 |          0.31 |        4 |           50   |          2.28 |     2.8  |     1    |         0.44 |        0.26 | 🟢 Positive |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | SUNPHARMA  |       1227.46 |          0.25 |        3 |           33.3 |          1.29 |     0    |     0.74 |         0.48 |        0.02 | 🟢 Positive |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | SBIN       |        990.88 |          0.2  |        2 |           50   |          2.42 |     0    |     0.86 |         0.32 |        0.02 | 🟢 Positive |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | RELIANCE   |        925.26 |          0.19 |        6 |           50   |          1.72 |     1.72 |     0.76 |         0.53 |        0.16 | 🟢 Positive |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | TATASTEEL  |        781.13 |          0.16 |        3 |           66.7 |          2.19 |     2.19 |     0.98 |         0.26 |        0.09 | 🟢 Positive |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | BHARTIARTL |        399.02 |          0.08 |        4 |           50   |          1.47 |     0    |     0.64 |         0.22 |        0.03 | 🟢 Positive |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | AXISBANK   |        306.96 |          0.06 |        3 |           33.3 |          1.55 |     1.57 |     0.39 |         0.17 |        0.04 | 🟢 Positive |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | BAJFINANCE |        176.78 |          0.04 |        2 |           50   |          1.79 |     1.79 |     0.42 |         0.1  |        0.02 | 🟢 Positive |
| alpha_18   | ALPHA_18_THREE_DAY_TREND_ORB | KOTAKBANK  |        167.76 |          0.03 |        1 |          100   |         99    |     0    |     0.24 |         0.22 |        0.01 | 🟢 Positive |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | ICICIBANK  |        160.49 |          0.03 |        3 |           33.3 |          2.88 |     4.79 |     0.47 |         0.05 |        0.03 | 🟢 Positive |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | MARUTI     |        113.92 |          0.02 |        1 |          100   |         99    |    99    |     0.61 |         0.03 |        0.01 | 🟢 Positive |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | SBIN       |         51.79 |          0.01 |        2 |           50   |          7.04 |     0    |     0.12 |         0.09 |        0    | 🟢 Positive |

## 3. Temporal Out-Of-Sample (OOS) Validation (30 Days Untouched Test Period)

| Alpha_ID   | Strategy                     |   IS_Lookback_Days |   IS_Trades |   IS_Net_PnL_INR |   IS_Basket_ROI_Pct |   OOS_Untouched_Days |   OOS_Trades |   OOS_Win_Rate_Pct |   OOS_Net_PnL_INR |   OOS_Basket_ROI_Pct |   OOS_Sharpe | OOS_Status                   |
|:-----------|:-----------------------------|-------------------:|------------:|-----------------:|--------------------:|---------------------:|-------------:|-------------------:|------------------:|---------------------:|-------------:|:-----------------------------|
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  |                 90 |         115 |        -11595.5  |               -0.17 |                   30 |            7 |               57.1 |           9957.1  |                 0.14 |         4.25 | 🟡 Insufficient Sample (<10T) |
| alpha_02   | ALPHA_02_AUCTION_ORB         |                 90 |         438 |        -96477.7  |               -1.38 |                   30 |          134 |               46.3 |           1981.66 |                 0.03 |         0.34 | 🟢 Strong Candidate (>=50T)   |
| alpha_03   | ALPHA_03_VWAP_REVERSION      |                 90 |          31 |         -3826.62 |               -0.05 |                   30 |           11 |               27.3 |          -3349.89 |                -0.05 |        -5.72 | 🔴 Negative OOS               |
| alpha_18   | ALPHA_18_THREE_DAY_TREND_ORB |                 90 |         183 |        -38754    |               -0.55 |                   30 |           56 |               25   |          -9609.88 |                -0.14 |        -4.1  | 🔴 Negative OOS               |

## 4. Full 2D Alpha × Asset Interaction Grid (4 Alphas × 14 Assets)

|                              | INFY                    | TCS                      | ICICIBANK               | HDFCBANK                 | SBIN                     | AXISBANK                 | KOTAKBANK                | RELIANCE                | LT                       | TATASTEEL                | BHARTIARTL              | BAJFINANCE              | MARUTI                   | SUNPHARMA                |
|:-----------------------------|:------------------------|:-------------------------|:------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:------------------------|:-------------------------|:-------------------------|:------------------------|:------------------------|:-------------------------|:-------------------------|
| ALPHA_02_AUCTION_ORB         | 🟢 +Rs 8.7k (38T|PF:1.4) | 🟢 +Rs 10.8k (40T|PF:1.7) | 🔴 -Rs 6.3k (43T|PF:0.8) | 🔴 -Rs 3.5k (31T|PF:0.8)  | 🔴 -Rs 14.9k (47T|PF:0.5) | 🔴 -Rs 20.6k (43T|PF:0.4) | 🔴 -Rs 28.9k (43T|PF:0.2) | 🟢 +Rs 5.9k (39T|PF:1.5) | 🔴 -Rs 16.8k (50T|PF:0.5) | 🔴 -Rs 18.8k (46T|PF:0.5) | 🟢 +Rs 5.5k (35T|PF:1.4) | 🟢 +Rs 3.2k (37T|PF:1.1) | 🔴 -Rs 10.0k (43T|PF:0.7) | 🔴 -Rs 15.0k (37T|PF:0.4) |
| ALPHA_03_VWAP_REVERSION      | 🔴 -Rs 2.6k (4T|PF:0.1)  | 🔴 -Rs 0.2k (4T|PF:0.9)   | 🟢 +Rs 0.2k (3T|PF:2.9)  | 🔴 -Rs 0.2k (1T|PF:0.0)   | 🟢 +Rs 0.1k (2T|PF:7.0)   | 🟢 +Rs 0.3k (3T|PF:1.6)   | 🔴 -Rs 2.7k (5T|PF:0.2)   | 🔴 -Rs 2.0k (4T|PF:0.1)  | 🔴 -Rs 1.0k (2T|PF:0.0)   | 🟢 +Rs 0.8k (3T|PF:2.2)   | 🟢 +Rs 0.4k (4T|PF:1.5)  | 🟢 +Rs 0.2k (2T|PF:1.8)  | 🟢 +Rs 0.1k (1T|PF:99.0)  | 🔴 -Rs 0.6k (4T|PF:0.7)   |
| ALPHA_14_GAP_MOMENTUM_DRIFT  | 🟢 +Rs 7.2k (6T|PF:4.0)  | 🟢 +Rs 2.8k (13T|PF:0.8)  | 0 Trades                | 🟢 +Rs 1.5k (4T|PF:2.3)   | 🟢 +Rs 1.0k (2T|PF:2.4)   | 🔴 -Rs 4.5k (15T|PF:0.3)  | 0 Trades                 | 🟢 +Rs 0.9k (6T|PF:1.7)  | 🔴 -Rs 2.1k (4T|PF:0.0)   | 🔴 -Rs 6.2k (5T|PF:0.0)   | 🟢 +Rs 2.4k (12T|PF:1.6) | 🔴 -Rs 3.6k (20T|PF:0.8) | 🔴 -Rs 2.2k (32T|PF:0.7)  | 🟢 +Rs 1.2k (3T|PF:1.3)   |
| ALPHA_18_THREE_DAY_TREND_ORB | 🔴 -Rs 0.8k (19T|PF:0.9) | 0 Trades                 | 🔴 -Rs 2.1k (12T|PF:0.5) | 🔴 -Rs 11.6k (48T|PF:0.4) | 🔴 -Rs 0.1k (6T|PF:1.0)   | 🔴 -Rs 3.8k (36T|PF:0.7)  | 🟢 +Rs 0.2k (1T|PF:99.0)  | 🔴 -Rs 3.9k (25T|PF:0.6) | 🔴 -Rs 5.0k (19T|PF:0.2)  | 🔴 -Rs 1.8k (8T|PF:0.4)   | 🔴 -Rs 3.1k (18T|PF:0.4) | 🔴 -Rs 8.0k (21T|PF:0.1) | 🔴 -Rs 6.2k (19T|PF:0.0)  | 🔴 -Rs 2.1k (7T|PF:0.0)   |

## 5. Inter-Alpha Daily Return Correlation Matrix (Cross-Strategy Redundancy)

|                              |   ALPHA_02_AUCTION_ORB |   ALPHA_03_VWAP_REVERSION |   ALPHA_14_GAP_MOMENTUM_DRIFT |   ALPHA_18_THREE_DAY_TREND_ORB |
|:-----------------------------|-----------------------:|--------------------------:|------------------------------:|-------------------------------:|
| ALPHA_02_AUCTION_ORB         |                   1    |                     -0.03 |                          0.25 |                           0.12 |
| ALPHA_03_VWAP_REVERSION      |                  -0.03 |                      1    |                         -0.05 |                          -0.01 |
| ALPHA_14_GAP_MOMENTUM_DRIFT  |                   0.25 |                     -0.05 |                          1    |                           0.01 |
| ALPHA_18_THREE_DAY_TREND_ORB |                   0.12 |                     -0.01 |                          0.01 |                           1    |

