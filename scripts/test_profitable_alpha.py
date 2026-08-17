"""
Ashva Multi-Asset Alpha Screener
Evaluates AlphaInstitutionalTrendPullback across 15m and hourly candles.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.data_lake import DataLake
from src.strategies.alpha_trend_pullback import AlphaInstitutionalTrendPullback
from src.backtest.engine import BacktestEngine
from src.analytics.tearsheet import QuantTearsheetGenerator


def main():
    lake = DataLake()
    strat = AlphaInstitutionalTrendPullback()
    tearsheet_gen = QuantTearsheetGenerator()

    print("=" * 90)
    print("[*] ASHVA ASYMMETRIC TREND PULLBACK ALPHA BACKTEST (15-MIN BARS, NET OF INDIAN TAXES)")
    print("=" * 90)

    for sym in ["INFY", "TCS", "RELIANCE", "ICICIBANK", "NIFTYBEES"]:
        df = lake.load_bars(sym, "15m")
        if df.empty:
            continue

        signals_df = strat.generate_signals(df)
        engine = BacktestEngine(initial_capital=500000.0)
        res = engine.run(signals_df, symbol=sym, strategy_id=strat.strategy_id)
        s = res.summary()
        
        status = "[PROFITABLE]" if s["total_net_pnl"] > 0 else "[LOSS]"
        print(f"{status:12s} {sym:10s} | Net PnL: Rs {s['total_net_pnl']:>10,.2f} | Net ROI: {s['net_roi_pct']:>6.2f}% | WinRate: {s['win_rate_pct']:>5.1f}% | Net PF: {s['net_profit_factor']:>4.2f} | Trades: {s['total_trades']:>2d} | Sharpe: {res.sharpe_ratio:>5.2f}")
        
        if s["total_net_pnl"] > 0:
            path = tearsheet_gen.generate_html_tearsheet(res)
            print(f"             +---> Tearsheet generated: {path}")

    print("\n" + "=" * 90)
    print("[*] ASHVA ML META-LABELED ASYMMETRIC TREND PULLBACK (15-MIN BARS, NET OF INDIAN TAXES)")
    print("=" * 90)

    from src.strategies.alpha_meta import AlphaMetaLabeledStrategy

    for sym in ["INFY", "ICICIBANK", "TCS", "RELIANCE"]:
        df = lake.load_bars(sym, "15m")
        if df.empty:
            continue

        primary = AlphaInstitutionalTrendPullback()
        meta_strat = AlphaMetaLabeledStrategy(primary_strategy=primary, parameters={"min_conviction_threshold": 0.50})
        meta_strat.fit_meta_model(df)
        signals_df = meta_strat.generate_signals(df)

        engine = BacktestEngine(initial_capital=500000.0)
        res = engine.run(signals_df, symbol=sym, strategy_id=meta_strat.strategy_id)
        s = res.summary()

        status = "[PROFITABLE]" if s["total_net_pnl"] > 0 else "[LOSS]"
        print(f"{status:12s} {sym:10s} | Net PnL: Rs {s['total_net_pnl']:>10,.2f} | Net ROI: {s['net_roi_pct']:>6.2f}% | WinRate: {s['win_rate_pct']:>5.1f}% | Net PF: {s['net_profit_factor']:>4.2f} | Trades: {s['total_trades']:>2d} | Sharpe: {res.sharpe_ratio:>5.2f} | MaxDD: {res.max_drawdown_pct:>4.2f}%")

        if s["total_net_pnl"] > 0:
            path = tearsheet_gen.generate_html_tearsheet(res)
            print(f"             +---> Tearsheet generated: {path}")

    print("=" * 90)



if __name__ == "__main__":
    main()
