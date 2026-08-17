"""
Unit Tests for Ashva Indian Market Regulatory Cost Engine
"""

import pytest
from src.analytics.indian_costs import IndianCostModel, Segment, Exchange, TradeCostBreakdown


def test_brokerage_cap():
    cost_model = IndianCostModel(brokerage_per_order=20.0, brokerage_pct_cap=0.0003)
    
    # Small trade: 10 shares at Rs 100 = Rs 1000. 0.03% = Rs 0.30 (lower than Rs 20)
    small_brokerage = cost_model.calculate_brokerage(1000.0)
    assert pytest.approx(small_brokerage, 0.01) == 0.30
    
    # Large trade: 1000 shares at Rs 2500 = Rs 25,00,000. 0.03% = Rs 750 (capped at Rs 20)
    large_brokerage = cost_model.calculate_brokerage(2500000.0)
    assert large_brokerage == 20.0


def test_intraday_roundtrip_cost_calculation():
    cost_model = IndianCostModel()
    
    # Buy 100 shares of Reliance at 2500, Sell at 2550
    # Buy Turnover = 2,50,000, Sell Turnover = 2,55,000, Gross PnL = +5,000
    breakdown = cost_model.calculate_trade_costs(
        buy_price=2500.0,
        sell_price=2550.0,
        quantity=100,
        segment=Segment.EQUITY_INTRADAY,
        slippage_bps=3.0,
    )

    assert breakdown.buy_turnover == 250000.0
    assert breakdown.sell_turnover == 255000.0
    assert breakdown.gross_pnl == 5000.0
    assert breakdown.brokerage == 40.0  # Rs 20 buy + Rs 20 sell
    assert breakdown.stt == round(255000.0 * 0.00025)  # 0.025% on sell only = ~64
    assert breakdown.stamp_duty == round(250000.0 * 0.00003)  # 0.003% on buy only = ~8
    assert breakdown.net_pnl < breakdown.gross_pnl
    assert breakdown.net_pnl > 4500.0  # Profitable trade after all deductions
    assert breakdown.net_return_pct > 0.0


def test_delivery_roundtrip_cost_calculation():
    cost_model = IndianCostModel()
    
    # Delivery has 0.1% STT on both buy and sell
    breakdown = cost_model.calculate_trade_costs(
        buy_price=1000.0,
        sell_price=1100.0,
        quantity=50,
        segment=Segment.EQUITY_DELIVERY,
        slippage_bps=0.0,
    )

    expected_stt = round(50000.0 * 0.001 + 55000.0 * 0.001)  # 50 + 55 = 105
    assert breakdown.stt == float(expected_stt)
    assert breakdown.stamp_duty == round(50000.0 * 0.00015)  # 0.015% on buy = ~8


def test_invalid_parameters():
    cost_model = IndianCostModel()
    
    with pytest.raises(ValueError):
        cost_model.calculate_trade_costs(buy_price=-10.0, sell_price=100.0, quantity=10)

    with pytest.raises(ValueError):
        cost_model.calculate_trade_costs(buy_price=100.0, sell_price=100.0, quantity=0)
