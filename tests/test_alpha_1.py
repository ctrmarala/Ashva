"""
Comprehensive Unit Tests for Alpha 1: Opening Gap Continuation
Validates contract metadata, discovery, dynamic universe, long/short signal mechanics,
strict zero look-ahead bias, exit rules, cost integration, and UI DAL hydration.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.strategies.alpha_001_opening_gap_continuation import Alpha1OpeningGapContinuation
from src.strategies.registry import get_all_strategies, get_strategy_by_name
from src.core.universe_manager import get_universe_symbols
from src.backtest.engine import BacktestEngine
from src.analytics.indian_costs import IndianCostModel, Segment
from src.ui.data_access import UIDataAccess


def create_synthetic_gap_df(gap_type: str = "BULLISH_CONTINUATION") -> pd.DataFrame:
    """
    Creates deterministic 2-day 15m OHLCV synthetic data to test gap mechanics:
    Day 1: 09:15 to 15:15 (Base close = 1000.0)
    Day 2: 09:15 to 15:15
    """
    timestamps = []
    # Day 1: 25 bars (09:15 to 15:15)
    d1 = datetime(2026, 8, 27)
    for i in range(25):
        timestamps.append(d1.replace(hour=9, minute=15) + timedelta(minutes=15 * i))
        
    # Day 2: 25 bars (09:15 to 15:15)
    d2 = datetime(2026, 8, 28)
    for i in range(25):
        timestamps.append(d2.replace(hour=9, minute=15) + timedelta(minutes=15 * i))

    total_bars = len(timestamps)
    opens = np.ones(total_bars) * 1000.0
    highs = np.ones(total_bars) * 1005.0
    lows = np.ones(total_bars) * 995.0
    closes = np.ones(total_bars) * 1000.0
    volumes = np.ones(total_bars) * 10000.0

    if gap_type == "BULLISH_CONTINUATION":
        # Day 2 opens at 1010.0 (+1.0% gap up vs Day 1 close 1000.0)
        # Bar 25 (Day 2 09:15): Open 1010, High 1015, Low 1008, Close 1014 (Bullish acceptance)
        opens[25] = 1010.0
        highs[25] = 1015.0
        lows[25] = 1008.0
        closes[25] = 1014.0
        
        # Bar 26 (Day 2 09:30): Open 1014, High 1020, Low 1012, Close 1018 (> Bar 25 High 1015 -> Trigger Long)
        opens[26] = 1014.0
        highs[26] = 1020.0
        lows[26] = 1012.0
        closes[26] = 1018.0

        # Subsequent bars trend higher (hitting take profit target)
        for j in range(27, total_bars):
            opens[j] = 1020.0 + (j - 26) * 2.0
            highs[j] = opens[j] + 5.0
            lows[j] = opens[j] - 1.0
            closes[j] = opens[j] + 4.0

    elif gap_type == "BEARISH_CONTINUATION":
        # Day 2 opens at 990.0 (-1.0% gap down vs Day 1 close 1000.0)
        # Bar 25 (Day 2 09:15): Open 990, High 992, Low 985, Close 986 (Bearish acceptance)
        opens[25] = 990.0
        highs[25] = 992.0
        lows[25] = 985.0
        closes[25] = 986.0
        
        # Bar 26 (Day 2 09:30): Open 986, High 988, Low 980, Close 982 (< Bar 25 Low 985 -> Trigger Short)
        opens[26] = 986.0
        highs[26] = 988.0
        lows[26] = 980.0
        closes[26] = 982.0

        for j in range(27, total_bars):
            opens[j] = 982.0
            highs[j] = 984.0
            lows[j] = 978.0
            closes[j] = 980.0

    elif gap_type == "REJECTED_GAP":
        # Gap up but opening bar collapses (close < open and low fills gap)
        opens[25] = 1010.0
        highs[25] = 1012.0
        lows[25] = 998.0
        closes[25] = 999.0
        for j in range(26, total_bars):
            opens[j] = 999.0
            highs[j] = 1001.0
            lows[j] = 995.0
            closes[j] = 998.0

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=pd.DatetimeIndex(timestamps))
    return df


def test_alpha1_contract_and_metadata():
    strat = Alpha1OpeningGapContinuation()
    assert strat.strategy_id == "1_alpha"
    assert strat.hypothesis_id == "1_alpha"
    assert "Opening Gap Continuation" in strat.name
    assert strat.metadata.category == "OPENING_AUCTION"
    assert strat.metadata.horizon.value == "INTRADAY"
    assert strat.get_parameter_grid() is not None
    assert "gap_threshold_pct" in strat.get_parameter_grid()
    
    # Target instruments dynamically supplied by Factory
    strat_with_universe = Alpha1OpeningGapContinuation({"target_instruments": get_universe_symbols()})
    assert len(strat_with_universe.metadata.target_instruments) >= 70


def test_alpha1_dynamic_discovery():
    all_strats = get_all_strategies(reload=True)
    assert "Alpha1OpeningGapContinuation" in all_strats or "1_alpha" in all_strats
    cls_ref = get_strategy_by_name("1_alpha")
    assert cls_ref is not None
    inst = cls_ref()
    assert inst.strategy_id == "1_alpha"


def test_alpha1_bullish_continuation_signal():
    strat = Alpha1OpeningGapContinuation()
    df = create_synthetic_gap_df("BULLISH_CONTINUATION")
    sig_df = strat.generate_signals(df)
    
    assert "signal" in sig_df.columns
    # Bar 26 should trigger Long signal (+1.0)
    assert sig_df["signal"].iloc[26] == 1.0
    # Day 1 should have zero signals
    assert (sig_df["signal"].iloc[:25] == 0.0).all()


def test_alpha1_bearish_continuation_signal():
    strat = Alpha1OpeningGapContinuation()
    df = create_synthetic_gap_df("BEARISH_CONTINUATION")
    sig_df = strat.generate_signals(df)
    
    assert "signal" in sig_df.columns
    # Bar 26 should trigger Short signal (-1.0)
    assert sig_df["signal"].iloc[26] == -1.0


def test_alpha1_rejected_gap_no_signal():
    strat = Alpha1OpeningGapContinuation()
    df = create_synthetic_gap_df("REJECTED_GAP")
    sig_df = strat.generate_signals(df)
    
    # Should produce 0 signals because early session rejected the gap
    assert (sig_df["signal"] == 0.0).all()


def test_alpha1_strict_zero_lookahead_leakage():
    strat = Alpha1OpeningGapContinuation()
    df = create_synthetic_gap_df("BULLISH_CONTINUATION")
    
    # Generate baseline signals
    sig_base = strat.generate_signals(df.copy())
    
    # Truncate future bars and ensure signal at bar 26 is identical
    truncated_df = df.iloc[:27].copy()
    sig_trunc = strat.generate_signals(truncated_df)
    
    assert sig_trunc["signal"].iloc[26] == sig_base["signal"].iloc[26]
    
    # Perturb future bar (bar 30) and verify bar 26 signal is completely unaltered
    df_perturbed = df.copy()
    df_perturbed.iloc[30, df_perturbed.columns.get_loc("close")] = 50000.0
    sig_perturbed = strat.generate_signals(df_perturbed)
    
    assert sig_perturbed["signal"].iloc[26] == sig_base["signal"].iloc[26]


def test_alpha1_backtest_execution_and_costs():
    strat = Alpha1OpeningGapContinuation()
    df = create_synthetic_gap_df("BULLISH_CONTINUATION")
    sig_df = strat.generate_signals(df)
    
    cost_model = IndianCostModel()
    engine = BacktestEngine(cost_model=cost_model, initial_capital=500000.0, segment=Segment.EQUITY_INTRADAY)
    res = engine.run(sig_df, symbol="RELIANCE", strategy_id="1_alpha")
    
    assert res.total_trades >= 1
    assert res.winning_trades >= 1
    assert res.total_net_pnl > 0
    assert res.total_brokerage_paid > 0
    assert res.total_taxes_paid > 0


def test_alpha1_ui_data_access_integration():
    dal = UIDataAccess()
    summary = dal.get_alpha_factory_summary()
    assert summary["total_alphas"] >= 1
    assert "untested" in summary
    assert summary["proven"] + summary["failed"] + summary["untested"] == summary["total_alphas"]
    
    df_reg = dal.get_alpha_registry_table()
    assert not df_reg.empty
    row_1 = df_reg[df_reg["alpha_id"] == "1_alpha"]
    assert not row_1.empty
    assert row_1.iloc[0]["category"] == "OPENING_AUCTION"
    
    detail = dal.get_alpha_detail("1_alpha")
    assert detail["alpha_id"] == "1_alpha"
    assert "qualification_gates" in detail