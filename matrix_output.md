# ASHVA ALPHA RESEARCH MATRIX AUDIT — [DEV MODE]

> **[!] DEVELOPMENT RUN -- NOT FOR RESEARCH CONCLUSIONS**

> [!WARNING]
> **DEVELOPMENT RUN ONLY**: This run used a truncated lookback and a 4-alpha subset for fast execution feedback. Do NOT use these metrics for formal research conclusions or live capital allocation.

- **Execution Mode**: `DEV`
- **Universe**: `NIFTY14` (14 Assets)
- **Audited Strategies**: `1 Alphas` (alpha_31)
- **Historical Lookback**: `120 Days` (IS: `90d` | Untouched OOS: `30d`)
- **Capital Deployment**: `Rs 500,000/Asset` (Total Basket Capital = `Rs 7,000,000`)
- **Cost Model**: Indian Statutory Taxes (STT, Exchange, GST, SEBI, Stamp Duty) + 3.0 bps Slippage
- **Timestamp**: `2026-08-18 23:51:54 IST`

## 1. Strategy Summary Matrix (1 Alphas)

| Alpha_ID   | Strategy                           |   Net_PnL_INR |   Basket_ROI_Pct |   Ann_Return_Pct |   Sharpe |   Max_DD_Pct |   PF_Lookback |   PF_60d |   Trades |   Win_Rate_Pct |   Avg_Trade_INR |   Recency_Q | Observed_Positive_Cluster                                                   |   Inter_Alpha_Corr |   Trade_PnL_Corr | Strategy_Classification           |
|:-----------|:-----------------------------------|--------------:|-----------------:|-----------------:|---------:|-------------:|--------------:|---------:|---------:|---------------:|----------------:|------------:|:----------------------------------------------------------------------------|-------------------:|-----------------:|:----------------------------------|
| alpha_31   | ALPHA_31_FAILED_OPENING_DRIVE_FADE |      -11769.5 |            -0.17 |            -0.51 |    -2.22 |         0.17 |          0.23 |     0.02 |       41 |           34.1 |          -287.1 |        -0.1 | 🟢 5/14 Assets: RELIANCE (+Rs 1,929), BAJFINANCE (+Rs 849), MARUTI (+Rs 504) |                  0 |                0 | 🔍 Candidate Asset Edges (5 Pairs) |

## 2. Candidate Alpha × Asset Edges (Observed Positive Pairs)

| Alpha_ID   | Strategy                           | Symbol     |   Net_PnL_INR |   Net_ROI_Pct |   Trades |   Win_Rate_Pct |   PF_Lookback |   PF_60d |   Sharpe |   Max_DD_Pct |   Recency_Q | Status     |
|:-----------|:-----------------------------------|:-----------|--------------:|--------------:|---------:|---------------:|--------------:|---------:|---------:|-------------:|------------:|:-----------|
| alpha_31   | ALPHA_31_FAILED_OPENING_DRIVE_FADE | RELIANCE   |       1928.99 |          0.39 |        1 |            100 |         99    |    99    |     2.93 |         0.12 |        0.12 | 🟢 Positive |
| alpha_31   | ALPHA_31_FAILED_OPENING_DRIVE_FADE | BAJFINANCE |        849.09 |          0.17 |        1 |            100 |         99    |    99    |     1.04 |         0.26 |        0.05 | 🟢 Positive |
| alpha_31   | ALPHA_31_FAILED_OPENING_DRIVE_FADE | MARUTI     |        504.17 |          0.1  |        2 |             50 |          1.34 |     1.34 |     0.4  |         0.3  |        0.04 | 🟢 Positive |
| alpha_31   | ALPHA_31_FAILED_OPENING_DRIVE_FADE | TCS        |        268.26 |          0.05 |        1 |            100 |         99    |    99    |     0.45 |         0.11 |        0.02 | 🟢 Positive |
| alpha_31   | ALPHA_31_FAILED_OPENING_DRIVE_FADE | INFY       |         70.46 |          0.01 |        1 |            100 |         99    |    99    |     0.08 |         0.21 |        0.01 | 🟢 Positive |

## 3. Temporal Out-Of-Sample (OOS) Validation (30 Days Untouched Test Period)

| Alpha_ID   | Strategy                           |   IS_Lookback_Days |   IS_Trades |   IS_Net_PnL_INR |   IS_Basket_ROI_Pct |   OOS_Untouched_Days |   OOS_Trades |   OOS_Win_Rate_Pct |   OOS_Net_PnL_INR |   OOS_Basket_ROI_Pct |   OOS_Sharpe | OOS_Status     |
|:-----------|:-----------------------------------|-------------------:|------------:|-----------------:|--------------------:|---------------------:|-------------:|-------------------:|------------------:|---------------------:|-------------:|:---------------|
| alpha_31   | ALPHA_31_FAILED_OPENING_DRIVE_FADE |                 90 |          22 |         -5796.25 |               -0.08 |                   30 |           19 |               31.6 |          -5124.33 |                -0.07 |        -4.51 | 🔴 Negative OOS |

## 4. Full 2D Alpha × Asset Interaction Grid (1 Alphas × 14 Assets)

|                                    | INFY                    | TCS                     | ICICIBANK              | HDFCBANK               | SBIN                   | AXISBANK               | KOTAKBANK              | RELIANCE                | LT                     | TATASTEEL              | BHARTIARTL             | BAJFINANCE              | MARUTI                 | SUNPHARMA              |
|:-----------------------------------|:------------------------|:------------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:------------------------|:-----------------------|:-----------------------|:-----------------------|:------------------------|:-----------------------|:-----------------------|
| ALPHA_31_FAILED_OPENING_DRIVE_FADE | 🟢 +Rs 0.1k (1T|PF:99.0) | 🟢 +Rs 0.3k (1T|PF:99.0) | 🔴 -Rs 6.0k (9T|PF:0.2) | 🔴 -Rs 2.2k (5T|PF:0.0) | 🔴 -Rs 0.3k (1T|PF:0.0) | 🔴 -Rs 2.6k (3T|PF:0.0) | 🔴 -Rs 0.6k (1T|PF:0.0) | 🟢 +Rs 1.9k (1T|PF:99.0) | 🔴 -Rs 1.8k (8T|PF:0.2) | 🔴 -Rs 0.8k (2T|PF:0.3) | 🔴 -Rs 0.3k (5T|PF:0.8) | 🟢 +Rs 0.8k (1T|PF:99.0) | 🟢 +Rs 0.5k (2T|PF:1.3) | 🔴 -Rs 0.7k (1T|PF:0.0) |

