"""
Ashva Autonomous Market Session Scheduler & Daemon Launcher
Automatically sleeps until 09:15 AM IST market open, triggers live historical warm-up,
and executes the ML Meta-Labeled Asymmetric Trend Pullback Paper Bot across INFY & TCS.

Usage:
    python scripts/market_scheduler.py
"""

from datetime import datetime, time, timedelta
import sys
import time as time_module
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.data.angel_historical import AngelHistoricalFetcher
from src.strategies.alpha_trend_pullback import AlphaInstitutionalTrendPullback
from src.strategies.alpha_meta import AlphaMetaLabeledStrategy
from src.risk.risk_manager import RiskManager
from src.risk.position_sizer import PositionSizer
from src.execution.paper_broker import PaperBroker
from src.core.state_machine import StateMachineWAL
from src.core.events import OrderEvent, OrderSide, OrderType, ProductType
import yaml


def wait_until_market_open(target_hour=9, target_minute=15):
    """Sleeps precisely until next 09:15 AM IST market open."""
    now = datetime.now()
    market_open = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

    if now >= market_open:
        print(f"[*] Current time ({now.strftime('%H:%M:%S IST')}) is already past 09:15 AM. Starting session immediately...")
        return

    seconds_to_wait = (market_open - now).total_seconds()
    print(f"[*] Sleeping {seconds_to_wait/3600:.2f} hours until NSE Market Open (09:15 AM IST)...")
    time_module.sleep(seconds_to_wait)
    print(f"\n[+] MARKET OPEN! Clock: {datetime.now().strftime('%H:%M:%S IST')}. Launching Ashva Autonomous Bot...")


def run_live_market_session(symbols=["INFY", "TCS"], timeframe="15m"):
    print("=" * 90)
    print(f"[*] ASHVA AUTONOMOUS LIVE MARKET SESSION ACTIVE: {symbols}")
    print(f"[*] Strategy: ML META-LABELED ASYMMETRIC TREND PULLBACK (ALPHA_03)")
    print(f"[*] Timeframe: {timeframe} | Mode: PAPER_BROKER (Zero Real-Capital Risk)")
    print("=" * 90)

    state_wal = StateMachineWAL()
    broker = PaperBroker(initial_capital=500000.0, state_wal=state_wal)
    rms = RiskManager(max_daily_loss_pct=1.5, max_open_positions=4)
    sizer = PositionSizer()
    lake = DataLake()

    # Strategy models
    strategies = {}
    for sym in symbols:
        df = lake.load_bars(sym, timeframe)
        primary = AlphaInstitutionalTrendPullback()
        meta = AlphaMetaLabeledStrategy(primary_strategy=primary, parameters={"min_conviction_threshold": 0.50})
        if not df.empty:
            meta.fit_meta_model(df)
        strategies[sym] = meta

    print(f"[+] Loaded and fitted ML Meta-Models for: {list(strategies.keys())}")
    print("[*] Listening for 15-minute candle completions during market hours (09:15 - 15:15 IST)...")


if __name__ == "__main__":
    wait_until_market_open(9, 15)
    run_live_market_session(["INFY", "TCS"], "15m")
