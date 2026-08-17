"""
Ashva Real-Time Live Forward Paper Trading Engine
Runs during active NSE market hours (09:15 AM - 03:30 PM IST).
Fetches live market ticks/candles via Angel One SmartAPI, evaluates ML Meta-Labeled Alpha strategies,
checks RMS risk limits, places paper orders with exact Indian tax accounting, and updates the Web Dashboard.

Usage:
    python scripts/run_live_paper_session.py --symbols INFY TCS ICICIBANK RELIANCE --timeframe 15m
"""

import argparse
import asyncio
from datetime import datetime, time
import sys
import time as time_module
from pathlib import Path
import yaml
import pandas as pd

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.events import OrderEvent, OrderSide, OrderType, ProductType
from src.core.event_bus import AsyncEventBus
from src.core.state_machine import StateMachineWAL
from src.data.data_lake import DataLake
from src.data.angel_historical import AngelHistoricalFetcher
from src.risk.risk_manager import RiskManager
from src.risk.position_sizer import PositionSizer
from src.execution.paper_broker import PaperBroker

from src.strategies.alpha_trend_pullback import AlphaInstitutionalTrendPullback
from src.strategies.alpha_vol_squeeze import AlphaVolatilitySqueeze
from src.strategies.alpha_meta import AlphaMetaLabeledStrategy
from src.portfolio.strategy_selector import StrategySelector


class LiveForwardPaperEngine:
    """
    Orchestrates live market forward paper trading during NSE market hours.
    """

    def __init__(self, symbols, timeframe="15m", initial_capital=500000.0):
        self.symbols = symbols
        self.timeframe = timeframe
        self.initial_capital = initial_capital

        self.state_wal = StateMachineWAL()
        self.data_lake = DataLake()
        self.event_bus = AsyncEventBus()
        self.rms = RiskManager(max_daily_loss_pct=1.5, max_open_positions=4)
        self.sizer = PositionSizer()
        self.broker = PaperBroker(initial_capital=initial_capital, state_wal=self.state_wal)
        self.selector = StrategySelector(data_lake=self.data_lake)

        # Load Angel One Fetcher for live market data
        with open("config/angel_one.yaml", "r") as f:
            cfg = yaml.safe_load(f).get("smartapi", {})

        self.fetcher = AngelHistoricalFetcher(
            api_key=cfg.get("api_key"),
            client_code=cfg.get("client_code"),
            password=cfg.get("password"),
            totp_secret=cfg.get("totp_secret"),
            data_lake=self.data_lake,
        )

        self.token_map = {
            "INFY": "1594",
            "TCS": "11536",
            "ICICIBANK": "4963",
            "RELIANCE": "2885",
        }

        self.strategies = {}

    def warmup_and_fit_models(self):
        """Fetches latest candle history from Angel One and trains Meta-Labelers."""
        print("[*] Authenticating with Angel One SmartAPI for Live Market Ingestion...")
        self.fetcher.initialize_session()
        print("[+] Session Authenticated! [PASS]")

        now = datetime.now()
        to_date = now.strftime("%Y-%m-%d %H:%M")
        from_date = (now - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M")

        print(f"[*] Ingesting and warm-up training ML Alpha Models across: {self.symbols}...")
        for sym in self.symbols:
            token = self.token_map.get(sym.upper(), "1594")
            df = self.fetcher.fetch_and_store(
                symbol=sym,
                token=token,
                timeframe=self.timeframe,
                from_date=from_date,
                to_date=to_date,
                exchange="NSE",
            )
            
            # Select best alpha strategy for asset
            analysis = self.selector.analyze_asset_regime(df)
            rec = analysis["recommended_strategy"]

            if "SQUEEZE" in rec:
                primary = AlphaVolatilitySqueeze()
            else:
                primary = AlphaInstitutionalTrendPullback()

            meta = AlphaMetaLabeledStrategy(primary_strategy=primary, parameters={"min_conviction_threshold": 0.50})
            if not df.empty:
                meta.fit_meta_model(df)

            self.strategies[sym] = {
                "strategy": meta,
                "strategy_id": meta.strategy_id,
                "primary_id": primary.strategy_id,
                "df": df,
            }
            print(f"    - {sym:10s} -> Assigned: {primary.strategy_id} + ML Meta-Labeler ({len(df)} bars)")

    def evaluate_live_signals(self):
        """Polls current price bar, evaluates signal, executes order if triggered."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        print(f"\n--- [{now_str}] LIVE MARKET CYCLE EVALUATION ---")

        for sym, item in self.strategies.items():
            token = self.token_map.get(sym.upper(), "1594")
            now = datetime.now()
            to_date = now.strftime("%Y-%m-%d %H:%M")
            from_date = (now - pd.Timedelta(days=5)).strftime("%Y-%m-%d %H:%M")

            # Fetch fresh bars from Angel One
            try:
                df = self.fetcher.fetch_and_store(
                    symbol=sym,
                    token=token,
                    timeframe=self.timeframe,
                    from_date=from_date,
                    to_date=to_date,
                    exchange="NSE",
                )
            except Exception as e:
                df = item["df"]

            if df.empty:
                continue

            last_bar = df.iloc[-1]
            curr_price = float(last_bar["close"])
            bar_time = df.index[-1]

            # Update Paper Broker with live market price for MTM valuation
            self.broker.update_market_price(sym, curr_price)

            # Generate signal
            strat = item["strategy"]
            signals = strat.generate_signals(df)
            signal_val = float(signals["signal"].iloc[-1])

            open_pos = self.broker.open_positions.get(sym.upper())

            # Evaluate entry
            if signal_val != 0.0 and open_pos is None:
                side = OrderSide.BUY if signal_val > 0 else OrderSide.SELL
                qty = self.sizer.calculate_fixed_risk_quantity(
                    equity=self.broker.equity,
                    entry_price=curr_price,
                    stop_loss_price=curr_price * (0.985 if side == OrderSide.BUY else 1.015),
                )

                order = OrderEvent(
                    order_id=f"LIVE_ORD_{sym}_{int(time_module.time()*1000)}",
                    symbol=sym,
                    timestamp=bar_time,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=qty,
                    product_type=ProductType.INTRADAY,
                    tag=item["strategy_id"],
                )

                is_valid, reason = self.rms.validate_order(
                    order=order,
                    current_equity=self.broker.equity,
                    current_price=curr_price,
                    open_positions_count=len(self.broker.open_positions),
                    current_time=now,
                )

                if is_valid:
                    fill = self.broker.submit_order(order, current_price=curr_price)
                    print(f"  [+] [ORDER EXECUTED] {order.side.value:4s} {fill.quantity} {sym} @ Rs {fill.fill_price:.2f} | Tag: {order.tag}")
                else:
                    print(f"  [!] [RMS REJECTED] {sym}: {reason}")

            elif signal_val == 0.0 and open_pos is not None:
                # Exit position
                exit_side = OrderSide.SELL if open_pos["side"] == "LONG" else OrderSide.BUY
                order = OrderEvent(
                    order_id=f"LIVE_EXIT_{sym}_{int(time_module.time()*1000)}",
                    symbol=sym,
                    timestamp=bar_time,
                    side=exit_side,
                    order_type=OrderType.MARKET,
                    quantity=open_pos["quantity"],
                    product_type=ProductType.INTRADAY,
                    tag=item["strategy_id"],
                )
                fill = self.broker.submit_order(order, current_price=curr_price)
                print(f"  [+] [POSITION CLOSED] {exit_side.value:4s} {fill.quantity} {sym} @ Rs {fill.fill_price:.2f}")

            else:
                pos_status = f"{open_pos['side']} {open_pos['quantity']} shares" if open_pos else "FLAT (Cash)"
                print(f"  - {sym:10s} : Rs {curr_price:>7.2f} | Signal: {signal_val:>4.1f} | Position: {pos_status}")

        pnl = self.broker.equity - self.broker.initial_capital
        pnl_pct = (pnl / self.broker.initial_capital) * 100.0
        print(f"[*] Portfolio Status: Equity: Rs {self.broker.equity:>10,.2f} | Live P&L: Rs {pnl:>+8,.2f} ({pnl_pct:>+5.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Ashva Live Forward Paper Trading Engine")
    parser.add_argument("--symbols", nargs="+", default=["INFY", "TCS", "ICICIBANK", "RELIANCE"])
    parser.add_argument("--timeframe", type=str, default="15m")
    args = parser.parse_args()

    print("=" * 95)
    print("[*] ASHVA AUTONOMOUS LIVE FORWARD PAPER TRADING ENGINE (NSE MARKET HOURS)")
    print(f"[*] Universe     : {args.symbols}")
    print(f"[*] Timeframe    : {args.timeframe}")
    print(f"[*] Mode         : LIVE FORWARD PAPER BROKER (Zero Risk)")
    print(f"[*] Dashboard    : http://localhost:8080")
    print("=" * 95)

    engine = LiveForwardPaperEngine(symbols=args.symbols, timeframe=args.timeframe)
    engine.warmup_and_fit_models()

    print("\n[*] Engine Running in Continuous Live Streaming Mode (09:15 - 15:30 IST)...")
    try:
        while True:
            engine.evaluate_live_signals()
            time_module.sleep(15)
    except KeyboardInterrupt:
        print("\n[*] Engine stopped safely by user.")


if __name__ == "__main__":
    main()
