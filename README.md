# ASHVA: Quantitative Research & Algorithmic Trading Platform

```
███████╗███████╗██╗  ██╗██╗   ██╗ █████╗ 
██╔════╝██╔════╝██║  ██║██║   ██║██╔══██╗
███████╗███████╗███████║██║   ██║███████║
╚════██║╚════██║██╔══██║╚╗   ██╔╝██╔══██║
███████║███████║██║  ██║ ╚████╔╝ ██║  ██║
╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝
Personal Quantitative Research & Algorithmic Trading Platform
Market: National Stock Exchange of India (NSE) | Broker: Angel One SmartAPI
```

Ashva is a disciplined, multi-strategy quantitative research and execution platform engineered for Indian Equities and Derivatives (NSE). It provides end-to-end infrastructure for strategy formulation, next-bar execution backtesting, exact Indian regulatory tax accounting (STT, GST, Stamp Duty, SEBI, Brokerage), risk management circuit breakers, and Angel One SmartAPI live order routing.

---

## ⚡ Quickstart Guide

### 1. Setup & Environment
Activate the pre-configured virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Run the complete 41-test verification test suite:
```powershell
pytest -v
```

---

### 2. Download Historical Market Data
Fetch and cache high-speed 5-minute / 15-minute / daily candles into the local DuckDB Data Lake:
```powershell
# Sync a single stock (e.g. RELIANCE)
python scripts/run_data_sync.py --symbol RELIANCE --timeframe 5m --period 1mo

# Sync entire default NIFTY 50 universe
python scripts/run_data_sync.py --universe --timeframe 5m --period 1mo
```

---

### 3. Quantitative Hypothesis Lab (Formulate, Test & Validate Alphas)
Run Marcos López de Prado's **Deflated Sharpe Ratio (DSR)**, **Combinatorial Purged Cross-Validation (CPCV)**, and **Monte Carlo 10,000-run Stress Tests** to filter out overfitted models:
```powershell
python scripts/run_hypothesis_lab.py --symbol RELIANCE --timeframe 5m --hypothesis orb
```

---

### 4. Train Deep Reinforcement Learning Agent
Train a continuous Actor-Critic / PPO policy network on the custom `AshvaTradingEnv` with Indian transaction friction:
```powershell
python scripts/run_rl_train.py --symbol RELIANCE --timeframe 5m --episodes 15
```

---

### 5. Run Real-Time Paper Trading Bot
Execute live tick-by-tick paper trading with full Write-Ahead Logging (WAL) and Risk Management (RMS) circuit breakers:
```powershell
python scripts/run_paper_bot.py --symbol RELIANCE --timeframe 5m
```

---

### 6. Launch Web Control Room Dashboard
Start the real-time telemetry server and open the interactive Dark-Theme Quant Dashboard:
```powershell
python src/ui/app.py
```
Open your browser at: **`http://localhost:8080`**

---

## 🛡️ SEBI Compliance & Angel One Static IP Setup

In compliance with SEBI regulatory circulars, Angel One SmartAPI mandates registering a **Static IPv4 address** for automated order placement:

1. Copy [`config/angel_one.example.yaml`](file:///c:/Work/Ashva/config/angel_one.example.yaml) to `config/angel_one.yaml`.
2. Fill in your `api_key`, `client_code`, `password`, `totp_secret`, and `whitelisted_static_ip`.
3. If running locally behind dynamic ISP IP, configure an egress static proxy in `proxy_url`.

---

## 📂 Architecture Overview

- **`src/core/`**: Asynchronous Event Bus (`asyncio`), typed Event dataclasses, and crash-resilient Write-Ahead Logging (WAL) State Machine.
- **`src/data/`**: DuckDB & Apache Parquet columnar Data Lake, Yahoo Finance loader, and Angel One SmartAPI historical fetcher.
- **`src/features/`**: Fractional Differentiation Engine ($\text{FFD}$), Anchored VWAP with $\pm 2\sigma$ dispersion bands, Volume Delta & CVD, Opening Range Volatility bands, Hurst Exponent regime classifier, and Financial NLP Sentiment Engine.
- **`src/research/`**: Hypothesis Factory, Triple-Barrier Dynamic Labeling, Deflated Sharpe Ratio (DSR), and Monte Carlo Stress Testing.
- **`src/strategies/`**: Institutional Opening Range Breakout (`AlphaInstitutionalORB`), Regime Mean Reversion (`AlphaRegimeAdaptiveMR`), ML Meta-Labeling (`AlphaMetaLabeledStrategy`), and Deep RL (`AlphaRLAgent`).
- **`src/portfolio/`**: Hierarchical Risk Parity (`HierarchicalRiskParityAllocator`).
- **`src/risk/`**: Risk Management System (`RiskManager`), Cornish-Fisher VaR & Expected Shortfall (CVaR), and Fractional Kelly Position Sizing.
- **`src/execution/`**: Institutional Paper Broker, Angel One SmartAPI Gateway, and Smart Order Router (TWAP & Iceberg).
- **`src/analytics/`**: Exact Indian Market Regulatory Cost Engine (`IndianCostModel` with STT, GST, Stamp Duty, Brokerage, Slippage) and Kolmogorov-Smirnov Alpha Drift Detector.
- **`src/ui/`**: Real-Time Web Control Room Dashboard.

---
*Built with quantitative rigor for the Indian Stock Exchange.*
