"""
Ashva Autonomous Paper Trading Runner
Connects EventBus, Alpha Strategy Engine, Risk Manager (RMS), and Paper Broker for live simulation.

Usage:
    python scripts/run_paper_bot.py --symbol INFY --timeframe 15m --strategy meta_pullback
"""

import argparse
import asyncio
from datetime import datetime
import sys
from pathlib import Path
import time

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.events import OrderEvent, OrderSide, OrderType, ProductType
from src.core.event_bus import AsyncEventBus
from src.core.state_machine import StateMachineWAL
from src.data.data_lake import DataLake
from src.risk.risk_manager import RiskManager
from src.risk.position_sizer import PositionSizer
from src.execution.paper_broker import PaperBroker

from src.strategies.alpha_orb import AlphaInstitutionalORB
from src.strategies.alpha_regime import AlphaRegimeAdaptiveMR
from src.strategies.alpha_trend_pullback import AlphaInstitutionalTrendPullback
from src.strategies.alpha_meta import AlphaMetaLabeledStrategy
from src.strategies.alpha_rl.agent import AlphaRLAgent


def get_strategy(strategy_name: str, df=None):
    strat_lower = strategy_name.lower()
    if strat_lower == "meta_pullback":
        primary = AlphaInstitutionalTrendPullback()
        meta = AlphaMetaLabeledStrategy(primary_strategy=primary, parameters={"min_conviction_threshold": 0.50})
        if df is not None and not df.empty:
            meta.fit_meta_model(df)
        return meta
    elif strat_lower == "pullback":
        return AlphaInstitutionalTrendPullback()
    elif strat_lower == "orb":
        return AlphaInstitutionalORB()
    elif strat_lower == "regime":
        return AlphaRegimeAdaptiveMR()
    elif strat_lower == "rl":
        return AlphaRLAgent()
    else:
        print(f"[!] Unknown strategy '{strategy_name}'. Defaulting to 'meta_pullback' (ML Meta-Labeled Trend Pullback)...")
        primary = AlphaInstitutionalTrendPullback()
        meta = AlphaMetaLabeledStrategy(primary_strategy=primary, parameters={"min_conviction_threshold": 0.50})
        if df is not None and not df.empty:
            meta.fit_meta_model(df)
        return meta


def main():
    parser = argparse.ArgumentParser(description="Ashva Autonomous Paper Trading Engine")
    parser.add_argument("--symbol", type=str, default="INFY", help="Symbol to run paper bot on")
    parser.add_argument("--timeframe", type=str, default="15m", help="Trading timeframe")
    parser.add_argument("--strategy", type=str, default="meta_pullback", 
                        choices=["meta_pullback", "pullback", "orb", "regime", "rl"],
                        help="Alpha Strategy to execute: meta_pullback (default), pullback, orb, regime, rl")

    args = parser.parse_args()

    print("=" * 90)
    print(f"[*] ASHVA AUTONOMOUS PAPER TRADING SYSTEM")
    print(f"[*] Target Asset : {args.symbol}")
    print(f"[*] Timeframe    : {args.timeframe}")
    print(f"[*] Strategy     : {args.strategy.upper()} (ML Meta-Labeled Asymmetric Trend Pullback)")
    print(f"[*] Mode         : PAPER_BROKER (Zero Real-Capital Risk)")
    print("=" * 90)

    # 1. Initialize Subsystems
    event_bus = AsyncEventBus()
    state_wal = StateMachineWAL()
    data_lake = DataLake()
    rms = RiskManager(max_daily_loss_pct=1.5, max_open_positions=4)
    position_sizer = PositionSizer()
    paper_broker = PaperBroker(initial_capital=500000.0, state_wal=state_wal)

    # 2. Ensure Data
    df = data_lake.load_bars(args.symbol, args.timeframe)
    if df.empty:
        print(f"[-] No market data found for {args.symbol} in DataLake. Please sync via Angel One SmartAPI first.")
        return

    print(f"[+] Loaded {len(df)} candles for simulation.")

    # 3. Instantiate Selected Strategy
    strategy = get_strategy(args.strategy, df=df)
    signals_df = strategy.generate_signals(df)

    print(f"[*] Starting Simulated Live Market Tick Streaming with {strategy.strategy_id}...")
    
    # 4. Stream Simulation Loop
    filled_orders_count = 0
    closed_trades_count = 0

    for i in range(len(signals_df)):
        bar_time = signals_df.index[i]
        curr_price = signals_df["close"].iloc[i]
        curr_signal = signals_df["signal"].iloc[i]

        paper_broker.update_market_price(args.symbol, curr_price)

        # Check existing open position
        open_pos = paper_broker.open_positions.get(args.symbol.upper())

        if curr_signal != 0.0 and open_pos is None:
            # Generate Entry Order
            side = OrderSide.BUY if curr_signal > 0 else OrderSide.SELL
            qty = position_sizer.calculate_fixed_risk_quantity(
                equity=paper_broker.equity,
                entry_price=curr_price,
                stop_loss_price=curr_price * (0.985 if side == OrderSide.BUY else 1.015),
            )

            order = OrderEvent(
                order_id=f"ORD_{args.symbol}_{int(time.time()*1000)}",
                symbol=args.symbol,
                timestamp=bar_time,
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
                product_type=ProductType.INTRADAY,
                tag=strategy.strategy_id,
            )

            # Validate with RMS
            is_valid, reason = rms.validate_order(
                order=order,
                current_equity=paper_broker.equity,
                current_price=curr_price,
                open_positions_count=len(paper_broker.open_positions),
                current_time=bar_time,
            )

            if is_valid:
                fill = paper_broker.submit_order(order, current_price=curr_price)
                filled_orders_count += 1
                print(f"[{str(bar_time)[:16]}] [ORDER FILLED] {order.side.value:4s} {fill.quantity:>3d} {args.symbol:8s} @ Rs {fill.fill_price:>7.2f} | Equity: Rs {paper_broker.equity:>10,.2f}")

        elif curr_signal == 0.0 and open_pos is not None:
            # Generate Exit Order
            exit_side = OrderSide.SELL if open_pos["side"] == "LONG" else OrderSide.BUY
            order = OrderEvent(
                order_id=f"ORD_EXIT_{args.symbol}_{int(time.time()*1000)}",
                symbol=args.symbol,
                timestamp=bar_time,
                side=exit_side,
                order_type=OrderType.MARKET,
                quantity=open_pos["quantity"],
                product_type=ProductType.INTRADAY,
                tag=strategy.strategy_id,
            )
            fill = paper_broker.submit_order(order, current_price=curr_price)
            closed_trades_count += 1
            print(f"[{str(bar_time)[:16]}] [POSITION CLOSED] {exit_side.value:4s} {fill.quantity:>3d} {args.symbol:8s} @ Rs {fill.fill_price:>7.2f} | Equity: Rs {paper_broker.equity:>10,.2f}")

    # 5. Summary
    pnl = paper_broker.equity - paper_broker.initial_capital
    roi = (pnl / paper_broker.initial_capital) * 100.0
    print("\n" + "=" * 90)
    print(f"[*] PAPER TRADING SIMULATION COMPLETED: {strategy.strategy_id}")
    print(f"  Initial Capital        : Rs {paper_broker.initial_capital:,.2f}")
    print(f"  Final Portfolio Equity : Rs {paper_broker.equity:,.2f}")
    print(f"  Net Total P&L          : Rs {pnl:+,.2f} ({roi:+.2f}%)")
    print(f"  Trades Closed          : {closed_trades_count}")
    print("=" * 90)


if __name__ == "__main__":
    main()
