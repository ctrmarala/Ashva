"""
Ashva Volatility Squeeze Strategy Screener
Evaluates AlphaVolatilitySqueeze and its ML Meta-Labeled Ensemble on official 15m Angel One Data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.strategies.alpha_vol_squeeze import AlphaVolatilitySqueeze
from src.strategies.alpha_meta import AlphaMetaLabeledStrategy
from src.backtest.engine import BacktestEngine
from src.analytics.tearsheet import QuantTearsheetGenerator


def main():
    lake = DataLake()
    tearsheet_gen = QuantTearsheetGenerator()

    print("=" * 90)
    print("[*] ASHVA VOLATILITY SQUEEZE STRATEGY BACKTEST (15-MIN BARS, NET OF INDIAN TAXES)")
    print("=" * 90)

    for sym in ["INFY", "TCS", "ICICIBANK", "RELIANCE"]:
        df = lake.load_bars(sym, "15m")
        if df.empty:
            continue

        # 1. Raw Squeeze Strategy
        raw_strat = AlphaVolatilitySqueeze()
        sig_raw = raw_strat.generate_signals(df)
        engine_raw = BacktestEngine(initial_capital=500000.0)
        res_raw = engine_raw.run(sig_raw, symbol=sym, strategy_id=raw_strat.strategy_id)
        s_raw = res_raw.summary()

        status_raw = "[PROFITABLE]" if s_raw["total_net_pnl"] > 0 else "[LOSS]"
        print(f"RAW   {status_raw:12s} {sym:10s} | Net PnL: Rs {s_raw['total_net_pnl']:>10,.2f} | Net ROI: {s_raw['net_roi_pct']:>6.2f}% | WinRate: {s_raw['win_rate_pct']:>5.1f}% | Net PF: {s_raw['net_profit_factor']:>4.2f} | Trades: {s_raw['total_trades']:>2d}")

        # 2. ML Meta-Labeled Squeeze Strategy
        meta_strat = AlphaMetaLabeledStrategy(primary_strategy=raw_strat, parameters={"min_conviction_threshold": 0.50})
        meta_strat.fit_meta_model(df)
        sig_meta = meta_strat.generate_signals(df)
        engine_meta = BacktestEngine(initial_capital=500000.0)
        res_meta = engine_meta.run(sig_meta, symbol=sym, strategy_id=meta_strat.strategy_id)
        s_meta = res_meta.summary()

        status_meta = "[PROFITABLE]" if s_meta["total_net_pnl"] > 0 else "[LOSS]"
        print(f"META  {status_meta:12s} {sym:10s} | Net PnL: Rs {s_meta['total_net_pnl']:>10,.2f} | Net ROI: {s_meta['net_roi_pct']:>6.2f}% | WinRate: {s_meta['win_rate_pct']:>5.1f}% | Net PF: {s_meta['net_profit_factor']:>4.2f} | Trades: {s_meta['total_trades']:>2d} | Sharpe: {res_meta.sharpe_ratio:>5.2f}")

        if s_meta["total_net_pnl"] > 0:
            path = tearsheet_gen.generate_html_tearsheet(res_meta)
            print(f"             +---> Tearsheet: {path}")

    print("=" * 90)


if __name__ == "__main__":
    main()
