import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.research.validator import StatisticalValidator
from src.research.hypothesis_factory import HypothesisFactory
from src.data.data_lake import DataLake
from src.backtest.engine import BacktestEngine
from src.analytics.indian_costs import Segment

# Import alphas
from src.strategies.alpha_03_vwap_reversion import Alpha03VWAPReversion
from src.strategies.alpha_04_gap_and_go import Alpha04GapAndGo
from src.strategies.alpha_11_donchian_breakout import Alpha11DonchianBreakout
from src.strategies.alpha_14_gap_momentum_drift import Alpha14GapMomentumDrift

def generate_matrix():
    print("=" * 80)
    print("[*] ASHVA INSTITUTIONAL ALPHA RESEARCH MATRIX AUDIT")
    print("=" * 80)
    
    lake = DataLake()
    # We will test on a proxy asset or the actual universe
    # For speed of the audit script, let's use a single liquid asset over the last 540 days.
    df = lake.load_bars("RELIANCE", "15m")
    if df.empty:
        print("[!] No data available for RELIANCE. Using random walk for testing.")
        dates = pd.date_range("2024-01-01", "2025-06-30", freq="15min")
        market_hours = (dates.hour >= 9) & (dates.hour <= 15)
        dates = dates[market_hours]
        returns = np.random.normal(0, 0.001, len(dates))
        prices = 2500 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.001, "low": prices * 0.999,
            "close": prices * (1 + np.random.normal(0, 0.0005, len(dates))),
            "volume": np.random.randint(1000, 10000, len(dates))
        }, index=dates)

    validator = StatisticalValidator()
    
    alphas_to_test = [
        Alpha03VWAPReversion(),
        Alpha04GapAndGo(),
        Alpha11DonchianBreakout(),
        Alpha14GapMomentumDrift()
    ]
    
    results = []
    daily_returns_dict = {}
    
    print(f"[*] Running full revalidation on {len(alphas_to_test)} Alphas...")
    
    for alpha in alphas_to_test:
        alpha_id = alpha.metadata.name
        print(f"    -> Validating {alpha_id}...")
        
        # We manually run a backtest to capture exact metrics and daily returns
        try:
            signals = alpha.generate_signals(df)
            engine = BacktestEngine(initial_capital=500000.0)
            res = engine.run(signals, symbol="RELIANCE", capital_per_trade_pct=0.50)
            
            daily_equity = res.equity_curve.resample("1D").last().dropna()
            daily_returns = daily_equity.pct_change().dropna()
            daily_returns_dict[alpha_id] = daily_returns
            
            days_in_market = (daily_equity.index[-1] - daily_equity.index[0]).days if len(daily_equity) > 1 else 1
            ann_return = (res.total_net_pnl / 500000.0) * (365.25 / max(1, days_in_market)) * 100
            net_roi = (res.total_net_pnl / 500000.0) * 100
            avg_trade = (res.total_net_pnl / max(1, res.total_trades))
            
            results.append({
                "Alpha": alpha_id,
                "Net ROI (%)": round(net_roi, 2),
                "Ann. Return (%)": round(ann_return, 2),
                "Sharpe": round(res.sharpe_ratio, 2),
                "Max DD (%)": round(res.max_drawdown_pct, 2),
                "Profit Factor": round(res.net_profit_factor, 2),
                "Trades": res.total_trades,
                "Win Rate (%)": round(res.win_rate_pct, 1),
                "Avg Trade": round(avg_trade, 2),
            })
        except Exception as e:
            print(f"       [!] Failed to evaluate {alpha_id}: {e}")
            
    if not results:
        return
        
    matrix_df = pd.DataFrame(results)
    
    # Compute Cross-Correlation Matrix
    returns_df = pd.DataFrame(daily_returns_dict).fillna(0)
    corr_matrix = returns_df.corr()
    
    # Get max correlation with ANY other alpha
    max_corrs = []
    for col in corr_matrix.columns:
        others = corr_matrix[col].drop(col)
        max_corrs.append(round(others.max(), 2) if not others.empty else 0.0)
        
    matrix_df["Inter-Alpha Corr"] = max_corrs
    
    # Classification logic
    classifications = []
    for _, row in matrix_df.iterrows():
        if row["Sharpe"] > 1.5 and row["Net ROI (%)"] > 5 and row["Inter-Alpha Corr"] < 0.4:
            classifications.append("🟢 Core")
        elif row["Sharpe"] > 1.0 and row["Inter-Alpha Corr"] >= 0.5:
            classifications.append("🟠 Redundant")
        elif row["Sharpe"] > 0.8 and row["Inter-Alpha Corr"] < 0.2:
            classifications.append("🔵 Diversifier")
        elif row["Sharpe"] > 1.0:
            classifications.append("🟡 Promising")
        else:
            classifications.append("🔴 Failed")
            
    matrix_df["Status"] = classifications
    
    output_text = "\n" + "=" * 120 + "\n"
    output_text += "ALPHA RESEARCH MATRIX\n"
    output_text += "=" * 120 + "\n"
    output_text += matrix_df.to_markdown(index=False) + "\n"
    output_text += "=" * 120 + "\n\n"
    
    output_text += "[*] INTER-ALPHA CORRELATION MATRIX (Redundancy Check)\n"
    output_text += corr_matrix.round(2).to_markdown() + "\n"
    
    with open("matrix_output.md", "w", encoding="utf-8") as f:
        f.write(output_text)
        
    print("[+] Wrote matrix output to matrix_output.md")

if __name__ == "__main__":
    generate_matrix()
