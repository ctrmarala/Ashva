# ASHVA ALPHA RESEARCH MATRIX AUDIT — [DEV MODE]

> **[!] DEVELOPMENT RUN -- NOT FOR RESEARCH CONCLUSIONS**

> [!WARNING]
> **DEVELOPMENT RUN ONLY**: This run used a truncated 120-day lookback and a 4-alpha subset for fast execution feedback. Do NOT use these metrics for formal research conclusions or live capital allocation.

- **Execution Mode**: `DEV`
- **Universe**: `NIFTY14` (14 Assets)
- **Audited Strategies**: `4 Alphas` (alpha_02, alpha_03, alpha_14, alpha_18)
- **Historical Lookback**: `120 Days` | **Timeframe**: `15m`
- **Capital Deployment**: `Rs 500,000/Asset` (Total Basket Capital = `Rs 7,000,000`)
- **Cost Model**: Indian Statutory Taxes (STT, Exchange, GST, SEBI, Stamp Duty) + 3.0 bps Slippage
- **Timestamp**: `2026-08-18 19:55:43 IST`

## 1. Strategy Summary Matrix (4 Alphas)

| Alpha_ID   | Strategy                     |   Net_PnL_INR |   Basket_ROI_Pct |   Ann_Return_Pct |   Sharpe |   Max_DD_Pct |   PF_Lookback |   PF_60d |   Trades |   Win_Rate_Pct |   Avg_Trade_INR |   Recency_Q | Profitable_Asset_Cluster                                                       |   Inter_Alpha_Corr |   Trade_PnL_Corr | Strategy_Classification    |
|:-----------|:-----------------------------|--------------:|-----------------:|-----------------:|---------:|-------------:|--------------:|---------:|---------:|---------------:|----------------:|------------:|:-------------------------------------------------------------------------------|-------------------:|-----------------:|:---------------------------|
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  |       1585.56 |             0.02 |             0.07 |     0.56 |         0.05 |          0.57 |    50.2  |       39 |           51.3 |            40.7 |        0.01 | 🟢 5/14 Assets: SUNPHARMA (+Rs 2,257), INFY (+Rs 1,338), BHARTIARTL (+Rs 1,241) |               0.1  |             0.14 | 🟡 Promising (Universal)    |
| alpha_03   | ALPHA_03_VWAP_REVERSION      |      -7173.12 |            -0.1  |            -0.31 |    -2.36 |         0.1  |          1.2  |     0.08 |       42 |           42.9 |          -170.8 |       -0.04 | 🟢 7/14 Assets: TATASTEEL (+Rs 781), BHARTIARTL (+Rs 399), AXISBANK (+Rs 307)   |               0.13 |             0.34 | 🔍 Asset-Specific (7 Pairs) |
| alpha_18   | ALPHA_18_THREE_DAY_TREND_ORB |      -8815.76 |            -0.13 |            -0.38 |    -3.55 |         0.13 |          0.1  |     0.07 |       49 |           26.5 |          -179.9 |       -0.06 | 🟢 3/14 Assets: TATASTEEL (+Rs 1,308), BAJFINANCE (+Rs 802), INFY (+Rs 382)     |               0.13 |             0.34 | 🔍 Asset-Specific (3 Pairs) |
| alpha_02   | ALPHA_02_AUCTION_ORB         |    -100745    |            -1.44 |            -4.38 |    -4.22 |         1.49 |          0.72 |     0.83 |      572 |           40.4 |          -176.1 |       -0.25 | 🟢 5/14 Assets: TCS (+Rs 10,825), INFY (+Rs 8,726), RELIANCE (+Rs 5,862)        |               0.05 |             0.18 | 🔍 Asset-Specific (5 Pairs) |

## 2. Alpha × Asset Positive Edge Cluster Map (Verified Profitable Pairs)

| Alpha_ID   | Strategy                     | Symbol     |   Net_PnL_INR |   Net_ROI_Pct |   Trades |   Win_Rate_Pct |   PF_Lookback |   PF_60d |   Sharpe |   Max_DD_Pct |   Recency_Q | Status       |
|:-----------|:-----------------------------|:-----------|--------------:|--------------:|---------:|---------------:|--------------:|---------:|---------:|-------------:|------------:|:-------------|
| alpha_02   | ALPHA_02_AUCTION_ORB         | TCS        |      10825.3  |          2.17 |       40 |           50   |          1.73 |     1.68 |     2.19 |         1.05 |        0.93 | 🟢 Profitable |
| alpha_02   | ALPHA_02_AUCTION_ORB         | INFY       |       8725.91 |          1.75 |       38 |           57.9 |          1.45 |     1.74 |     1.61 |         1.52 |        0.92 | 🟢 Profitable |
| alpha_02   | ALPHA_02_AUCTION_ORB         | RELIANCE   |       5861.92 |          1.17 |       39 |           51.3 |          1.47 |     1.19 |     1.8  |         0.84 |        0.56 | 🟢 Profitable |
| alpha_02   | ALPHA_02_AUCTION_ORB         | BHARTIARTL |       5473.14 |          1.09 |       35 |           54.3 |          1.37 |     0.77 |     1.31 |         1.42 |        0.26 | 🟢 Profitable |
| alpha_02   | ALPHA_02_AUCTION_ORB         | BAJFINANCE |       3196.02 |          0.64 |       37 |           56.8 |          1.14 |     1.06 |     0.6  |         1.94 |        0.37 | 🟢 Profitable |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | SUNPHARMA  |       2257.29 |          0.45 |        2 |          100   |         99    |    99    |     1.78 |         0.02 |        0.15 | 🟢 Profitable |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | INFY       |       1338.39 |          0.27 |        5 |           60   |         12.54 |    99    |     1.79 |         0.05 |        0.13 | 🟢 Profitable |
| alpha_18   | ALPHA_18_THREE_DAY_TREND_ORB | TATASTEEL  |       1308.26 |          0.26 |        2 |           50   |         51.48 |     0    |     1.87 |         0.03 |        0.06 | 🟢 Profitable |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | BHARTIARTL |       1241.42 |          0.25 |        2 |          100   |         99    |    99    |     1.82 |         0.04 |        0.07 | 🟢 Profitable |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | TCS        |       1137.47 |          0.23 |        4 |           75   |         16.12 |    99    |     1.78 |         0.05 |        0.1  | 🟢 Profitable |
| alpha_14   | ALPHA_14_GAP_MOMENTUM_DRIFT  | SBIN       |       1058.4  |          0.21 |        2 |          100   |         99    |    99    |     1.59 |         0.03 |        0.05 | 🟢 Profitable |
| alpha_18   | ALPHA_18_THREE_DAY_TREND_ORB | BAJFINANCE |        802.4  |          0.16 |        4 |           50   |          2.97 |     5.74 |     1.43 |         0.1  |        0.12 | 🟢 Profitable |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | TATASTEEL  |        781.13 |          0.16 |        3 |           66.7 |          2.19 |     2.19 |     0.98 |         0.26 |        0.09 | 🟢 Profitable |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | BHARTIARTL |        399.02 |          0.08 |        4 |           50   |          1.47 |     0    |     0.64 |         0.22 |        0.03 | 🟢 Profitable |
| alpha_18   | ALPHA_18_THREE_DAY_TREND_ORB | INFY       |        381.69 |          0.08 |        3 |           66.7 |          6.88 |     2.05 |     1    |         0.05 |        0.03 | 🟢 Profitable |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | AXISBANK   |        306.96 |          0.06 |        3 |           33.3 |          1.55 |     1.57 |     0.39 |         0.17 |        0.04 | 🟢 Profitable |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | BAJFINANCE |        176.78 |          0.04 |        2 |           50   |          1.79 |     1.79 |     0.42 |         0.1  |        0.02 | 🟢 Profitable |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | ICICIBANK  |        160.49 |          0.03 |        3 |           33.3 |          2.88 |     4.79 |     0.47 |         0.05 |        0.03 | 🟢 Profitable |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | MARUTI     |        113.92 |          0.02 |        1 |          100   |         99    |    99    |     0.61 |         0.03 |        0.01 | 🟢 Profitable |
| alpha_03   | ALPHA_03_VWAP_REVERSION      | SBIN       |         51.79 |          0.01 |        2 |           50   |          7.04 |     0    |     0.12 |         0.09 |        0    | 🟢 Profitable |

## 3. Full 2D Alpha × Asset Interaction Grid (4 Alphas × 14 Assets)

|                              | INFY                    | TCS                      | ICICIBANK               | HDFCBANK                | SBIN                     | AXISBANK                 | KOTAKBANK                | RELIANCE                | LT                       | TATASTEEL                | BHARTIARTL              | BAJFINANCE              | MARUTI                   | SUNPHARMA                |
|:-----------------------------|:------------------------|:-------------------------|:------------------------|:------------------------|:-------------------------|:-------------------------|:-------------------------|:------------------------|:-------------------------|:-------------------------|:------------------------|:------------------------|:-------------------------|:-------------------------|
| ALPHA_02_AUCTION_ORB         | 🟢 +Rs 8.7k (38T|PF:1.4) | 🟢 +Rs 10.8k (40T|PF:1.7) | 🔴 -Rs 6.3k (43T|PF:0.8) | 🔴 -Rs 3.5k (31T|PF:0.8) | 🔴 -Rs 14.9k (47T|PF:0.5) | 🔴 -Rs 20.6k (43T|PF:0.4) | 🔴 -Rs 28.9k (43T|PF:0.2) | 🟢 +Rs 5.9k (39T|PF:1.5) | 🔴 -Rs 16.8k (50T|PF:0.5) | 🔴 -Rs 18.8k (46T|PF:0.5) | 🟢 +Rs 5.5k (35T|PF:1.4) | 🟢 +Rs 3.2k (37T|PF:1.1) | 🔴 -Rs 10.0k (43T|PF:0.7) | 🔴 -Rs 15.0k (37T|PF:0.4) |
| ALPHA_03_VWAP_REVERSION      | 🔴 -Rs 2.6k (4T|PF:0.1)  | 🔴 -Rs 0.2k (4T|PF:0.9)   | 🟢 +Rs 0.2k (3T|PF:2.9)  | 🔴 -Rs 0.2k (1T|PF:0.0)  | 🟢 +Rs 0.1k (2T|PF:7.0)   | 🟢 +Rs 0.3k (3T|PF:1.6)   | 🔴 -Rs 2.7k (5T|PF:0.2)   | 🔴 -Rs 2.0k (4T|PF:0.1)  | 🔴 -Rs 1.0k (2T|PF:0.0)   | 🟢 +Rs 0.8k (3T|PF:2.2)   | 🟢 +Rs 0.4k (4T|PF:1.5)  | 🟢 +Rs 0.2k (2T|PF:1.8)  | 🟢 +Rs 0.1k (1T|PF:99.0)  | 🔴 -Rs 0.6k (4T|PF:0.7)   |
| ALPHA_14_GAP_MOMENTUM_DRIFT  | 🟢 +Rs 1.3k (5T|PF:12.5) | 🟢 +Rs 1.1k (4T|PF:16.1)  | 0 Trades                | 🔴 -Rs 0.5k (3T|PF:0.5)  | 🟢 +Rs 1.1k (2T|PF:99.0)  | 🔴 -Rs 0.4k (2T|PF:0.0)   | 0 Trades                 | 🔴 -Rs 0.6k (3T|PF:0.3)  | 🔴 -Rs 0.5k (2T|PF:0.1)   | 🔴 -Rs 0.9k (4T|PF:0.4)   | 🟢 +Rs 1.2k (2T|PF:99.0) | 🔴 -Rs 1.8k (4T|PF:0.2)  | 🔴 -Rs 0.8k (6T|PF:0.7)   | 🟢 +Rs 2.3k (2T|PF:99.0)  |
| ALPHA_18_THREE_DAY_TREND_ORB | 🟢 +Rs 0.4k (3T|PF:6.9)  | 0 Trades                 | 🔴 -Rs 0.4k (4T|PF:0.4)  | 🔴 -Rs 3.2k (10T|PF:0.1) | 🔴 -Rs 0.4k (2T|PF:0.1)   | 🔴 -Rs 1.0k (6T|PF:0.4)   | 🔴 -Rs 0.3k (1T|PF:0.0)   | 🔴 -Rs 1.8k (5T|PF:0.0)  | 🔴 -Rs 1.5k (4T|PF:0.1)   | 🟢 +Rs 1.3k (2T|PF:51.5)  | 🔴 -Rs 0.2k (3T|PF:0.3)  | 🟢 +Rs 0.8k (4T|PF:3.0)  | 🔴 -Rs 1.6k (4T|PF:0.0)   | 🔴 -Rs 1.0k (1T|PF:0.0)   |

## 4. Inter-Alpha Daily Return Correlation Matrix (Cross-Strategy Redundancy)

|                              |   ALPHA_02_AUCTION_ORB |   ALPHA_03_VWAP_REVERSION |   ALPHA_14_GAP_MOMENTUM_DRIFT |   ALPHA_18_THREE_DAY_TREND_ORB |
|:-----------------------------|-----------------------:|--------------------------:|------------------------------:|-------------------------------:|
| ALPHA_02_AUCTION_ORB         |                   1    |                     -0.03 |                         -0.05 |                          -0.03 |
| ALPHA_03_VWAP_REVERSION      |                  -0.03 |                      1    |                          0.01 |                          -0.13 |
| ALPHA_14_GAP_MOMENTUM_DRIFT  |                  -0.05 |                      0.01 |                          1    |                           0.1  |
| ALPHA_18_THREE_DAY_TREND_ORB |                  -0.03 |                     -0.13 |                          0.1  |                           1    |

