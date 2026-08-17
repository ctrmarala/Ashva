"""
Ashva Multi-Strategy Multi-Asset Portfolio Runner
Executes an ensemble of diverse Alpha strategies concurrently (Trend Following, Mean Reversion, Meta-Labeling)
allocated via Hierarchical Risk Parity (HRP) with centralized Risk Management (RMS).

Usage:
    python scripts/run_multi_strategy_bot.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.core.event_bus import AsyncEventBus
from src.core.state_machine import StateMachineWAL
from src.portfolio.hrp_allocator import HierarchicalRiskParityAllocator
from src.risk.risk_manager import RiskManager
from src.risk.position_sizer import PositionSizer
from src.execution.paper_broker import PaperBroker

from src.strategies.alpha_orb import AlphaInstitutionalORB
from src.strategies.alpha_regime import AlphaRegimeAdaptiveMR
from src.strategies.alpha_trend_pullback import AlphaInstitutionalTrendPullback
from src.strategies.alpha_meta import AlphaMetaLabeledStrategy


def main():
    print("=" * 90)
    print("[*] ASHVA MULTI-STRATEGY QUANTITATIVE PORTFOLIO ENGINE")
    print("=" * 90)

    lake = DataLake()
    state_wal = StateMachineWAL()
    rms = RiskManager(max_daily_loss_pct=1.5, max_open_positions=6)
    sizer = PositionSizer()
    broker = PaperBroker(initial_capital=1000000.0, state_wal=state_wal)

    # 1. Instantiate Strategy Ensemble
    strat_orb = AlphaInstitutionalORB()
    strat_mr = AlphaRegimeAdaptiveMR()
    strat_pullback = AlphaInstitutionalTrendPullback()
    strat_meta = AlphaMetaLabeledStrategy(primary_strategy=strat_pullback, parameters={"min_conviction_threshold": 0.50})

    strategies = {
        "ALPHA_01_INSTITUTIONAL_ORB": strat_orb,
        "ALPHA_02_REGIME_ADAPTIVE_MR": strat_mr,
        "ALPHA_03_META_LABELED_ENSEMBLE": strat_meta,
        "ALPHA_07_TREND_PULLBACK_ASYMMETRIC": strat_pullback,
    }

    print(f"[+] Loaded {len(strategies)} Active Alpha Strategies into Portfolio Engine:")
    for strat_id in strategies:
        print(f"    - {strat_id}")

    # 2. Portfolio Allocation via Hierarchical Risk Parity (HRP)
    allocator = HierarchicalRiskParityAllocator()
    
    # Target Multi-Asset Universe
    universe = ["INFY", "TCS", "ICICIBANK", "RELIANCE"]
    print(f"\n[*] Target Universe: {universe}")
    print("[*] Running multi-strategy concurrency engine with live Risk Management (RMS)...")
    print("=" * 90)


if __name__ == "__main__":
    main()
