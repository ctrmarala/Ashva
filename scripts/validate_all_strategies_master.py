"""
Ashva Master Quantitative Alpha Validation Suite
Backtests and validates all 4 Alpha Strategies across 14 Liquid NSE Blue Chips over 18 Months:
- Alpha 01: TrendSurfer (Dual EMA Pullback)
- Alpha 02: Auction ORB Pro (Opening Range Breakout)
- Alpha 03: VWAP Mean Reversion
- Alpha 04: Gap & Go (Institutional Continuation)
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.strategies.alpha_trend_surfer import AlphaTrendSurfer
from src.strategies.alpha_orb_pro import AlphaAuctionORBPro
from src.strategies.alpha_03_vwap_reversion import Alpha03VWAPReversion
from src.strategies.alpha_04_gap_and_go import Alpha04GapAndGo
from src.research.validator import StatisticalValidator
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.research.hypothesis import HypothesisStatus


def run_master_validation():
    print("=" * 115)
    print("[*] ASHVA RESEARCH LAB: MASTER QUANTITATIVE ALPHA BACKTEST & VALIDATION AUDIT")
    print("=" * 115)

    lake = DataLake(read_only=True)
    cost_model = IndianCostModel(default_slippage_bps=3.0)
    validator = StatisticalValidator(cost_model=cost_model)
    engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0)

    symbols = [
        "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
        "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
        "BAJFINANCE", "MARUTI", "SUNPHARMA"
    ]

    strategies = [
        ("ALPHA_01_TRENDSURFER", AlphaTrendSurfer()),
        ("ALPHA_02_AUCTION_ORB", AlphaAuctionORBPro()),
        ("ALPHA_03_VWAP_REVERSION", Alpha03VWAPReversion()),
        ("ALPHA_04_GAP_AND_GO", Alpha04GapAndGo()),
    ]

    master_results = []

    for strat_id, strat_obj in strategies:
        print(f"\n" + "#" * 115)
        print(f"[+] AUDITING STRATEGY: {strat_id}")
        print("#" * 115)

        total_strat_pnl = 0.0
        total_strat_trades = 0
        total_strat_wins = 0
        total_strat_taxes = 0.0
        stock_reports = []

        for sym in symbols:
            df = lake.load_bars(sym, "15m")
            if df.empty or len(df) < 200:
                continue

            signals_df = strat_obj.generate_signals(df)
            res = engine.run(signals_df, symbol=sym, strategy_id=strat_id, risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)

            total_strat_pnl += res.total_net_pnl
            total_strat_trades += res.total_trades
            total_strat_wins += res.winning_trades
            total_strat_taxes += res.total_taxes_paid

            # Classify status using calibrated validator rules
            if res.net_profit_factor < 1.0:
                verdict = "[REJECTED: Negative Net Expectancy]"
            elif res.total_trades < 25:
                verdict = "[LOW_FREQ_WATCH: N < 25, Positive Payoff]"
            elif res.net_profit_factor >= 1.08 and res.sharpe_ratio > 0:
                verdict = "[FORWARD_PAPER: Viable Edge]"
            else:
                verdict = "[RESEARCH_CANDIDATE]"

            stock_reports.append({
                "Strategy": strat_id,
                "Symbol": sym,
                "Net_PnL_INR": round(res.total_net_pnl, 2),
                "Net_ROI_Pct": round(res.net_roi_pct, 2),
                "Trades": res.total_trades,
                "Win_Rate_Pct": round(res.win_rate_pct, 1),
                "Net_Profit_Factor": round(res.net_profit_factor, 2) if res.net_profit_factor < 90 else 99.0,
                "Sharpe": round(res.sharpe_ratio, 2),
                "Max_DD_Pct": round(res.max_drawdown_pct, 2),
                "Total_Costs_INR": round(res.total_taxes_paid, 2),
                "Verdict": verdict,
            })

        strat_df = pd.DataFrame(stock_reports)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(strat_df.to_string(index=False))

        portfolio_wr = (total_strat_wins / total_strat_trades * 100.0) if total_strat_trades > 0 else 0.0
        print("-" * 115)
        print(f"[*] {strat_id} PORTFOLIO TOTAL: Net P&L = Rs {total_strat_pnl:+,.2f} | Trades = {total_strat_trades} | Win Rate = {portfolio_wr:.1f}% | Taxes Paid = Rs {total_strat_taxes:,.2f}")

        master_results.extend(stock_reports)

    # Master Comparative Summary
    print("\n" + "=" * 115)
    print("[*] MASTER INSTITUTIONAL LEADERBOARD (SORTED BY NET P&L)")
    print("=" * 115)
    master_df = pd.DataFrame(master_results)
    master_df_sorted = master_df.sort_values(by="Net_PnL_INR", ascending=False)
    print(master_df_sorted.to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    run_master_validation()
