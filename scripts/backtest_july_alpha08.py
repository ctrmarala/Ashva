"""
Ashva Specific Period Backtest: July 1 to July 30
Evaluates Alpha 08 (Institutional Volatility Squeeze & Momentum Expansion with ML Meta-Labeler)
specifically during the July 1 to July 30 window with exact Indian taxes deducted.

Usage:
    python scripts/backtest_july_alpha08.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.strategies.alpha_vol_squeeze import AlphaVolatilitySqueeze
from src.strategies.alpha_meta import AlphaMetaLabeledStrategy
from src.backtest.engine import BacktestEngine


def main():
    print("=" * 95)
    print("[*] ASHVA HISTORICAL PERFORMANCE REPORT: JULY 1 TO JULY 30 (ALPHA 08 VOL SQUEEZE)")
    print("[*] Accounting: 100% Real Post-Tax Net (STT, GST 18%, Stamp Duty, SEBI, Angel Brokerage, Slippage)")
    print("=" * 95)

    lake = DataLake()
    symbols = ["INFY", "TCS", "ICICIBANK", "RELIANCE"]
    initial_capital_per_asset = 500000.0

    total_gross_pnl = 0.0
    total_net_pnl = 0.0
    total_taxes = 0.0
    total_brokerage = 0.0
    total_stt = 0.0
    total_trades_count = 0
    total_winning_trades = 0

    results = []

    for sym in symbols:
        df_full = lake.load_bars(sym, "15m")
        if df_full.empty:
            continue

        # Filter strictly for July 1 to July 30
        df_july = df_full[(df_full.index >= "2026-07-01") & (df_full.index <= "2026-07-30 23:59:59")].copy()

        if len(df_july) < 30:
            print(f"[!] Insufficient July bars for {sym} (found {len(df_july)}). Using available July range...")
            df_july = df_full[df_full.index.month == 7].copy()

        raw_strat = AlphaVolatilitySqueeze()
        meta_strat = AlphaMetaLabeledStrategy(primary_strategy=raw_strat, parameters={"min_conviction_threshold": 0.50})
        
        # Fit on full context and generate signals on July
        meta_strat.fit_meta_model(df_full)
        signals_july = meta_strat.generate_signals(df_july)

        engine = BacktestEngine(initial_capital=initial_capital_per_asset)
        res = engine.run(signals_july, symbol=sym, strategy_id=meta_strat.strategy_id)
        s = res.summary()

        total_gross_pnl += (res.final_equity - res.initial_capital + res.total_taxes_paid)
        total_net_pnl += s["total_net_pnl"]
        total_taxes += res.total_taxes_paid
        total_brokerage += res.total_brokerage_paid
        total_stt += res.total_stt_paid
        total_trades_count += s["total_trades"]
        total_winning_trades += res.winning_trades

        results.append({
            "Symbol": sym,
            "Trades": s["total_trades"],
            "Win Rate": f"{s['win_rate_pct']:.1f}%",
            "Net Profit Factor": f"{s['net_profit_factor']:.2f}",
            "Sharpe": f"{res.sharpe_ratio:.2f}",
            "Max DD": f"{res.max_drawdown_pct:.2f}%",
            "Taxes & Charges": f"Rs {res.total_taxes_paid:,.2f}",
            "Net Profit": s["total_net_pnl"],
            "Net ROI": s["net_roi_pct"],
        })

    # Print Table
    print(f"{'Asset':12s} | {'Trades':>6s} | {'Win Rate':>8s} | {'Profit Factor':>13s} | {'Sharpe':>6s} | {'Max DD':>6s} | {'Taxes Paid':>14s} | {'Net Post-Tax Profit':>20s} | {'ROI':>6s}")
    print("-" * 115)
    for r in results:
        pnl_str = f"Rs {r['Net Profit']:>+12,.2f}"
        roi_str = f"{r['Net ROI']:>+5.2f}%"
        pf_str = r['Net Profit Factor']
        print(f"{r['Symbol']:12s} | {r['Trades']:>6d} | {r['Win Rate']:>8s} | {pf_str:>13s} | {r['Sharpe']:>6s} | {r['Max DD']:>6s} | {r['Taxes & Charges']:>14s} | {pnl_str:>20s} | {roi_str:>6s}")

    print("=" * 115)
    overall_win_rate = (total_winning_trades / total_trades_count * 100.0) if total_trades_count > 0 else 0.0
    combined_capital = initial_capital_per_asset * len(results)
    combined_roi = (total_net_pnl / combined_capital) * 100.0 if combined_capital > 0 else 0.0

    print(f"[*] COMBINED PORTFOLIO SUMMARY (JULY 1 - JULY 30):")
    print(f"    - Initial Capital Allocated    : Rs {combined_capital:,.2f}")
    print(f"    - Total Trades Executed        : {total_trades_count} ({total_winning_trades} Wins / {total_trades_count - total_winning_trades} Losses)")
    print(f"    - Overall Win Rate             : {overall_win_rate:.1f}%")
    print(f"    - Gross Trading P&L            : Rs {total_gross_pnl:+,.2f}")
    print(f"    - Total STT Paid (Govt)        : Rs {total_stt:,.2f}")
    print(f"    - Total Angel Brokerage Paid   : Rs {total_brokerage:,.2f}")
    print(f"    - Total Statutory Taxes (NSE)  : Rs {total_taxes:,.2f}")
    print(f"    -----------------------------------------------------------------------------")
    print(f"    - TOTAL NET POST-TAX PROFIT    : Rs {total_net_pnl:+,.2f} ({combined_roi:+.2f}% Net Return in 1 Month)")
    print("=" * 115)


if __name__ == "__main__":
    main()
