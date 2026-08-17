"""
Unit Tests for Position Sizer Engine
"""

import pytest
from src.risk.position_sizer import PositionSizer


def test_fixed_risk_quantity_sizing():
    sizer = PositionSizer(default_risk_per_trade_pct=1.0, max_capital_per_trade_pct=25.0)
    
    # Capital = 5,00,000, 1% Risk = Rs 5,000
    # Entry = 2500, Stop = 2450 (Risk per share = Rs 50)
    # Qty = 5000 / 50 = 100 shares
    # Value = 100 * 2500 = 2,50,000 (which is 50% capital -> capped at 25% max capital = 50 shares)
    qty = sizer.calculate_fixed_risk_quantity(
        equity=500000.0,
        entry_price=2500.0,
        stop_loss_price=2450.0,
        risk_pct=1.0,
    )

    # 25% of 5,00,000 = 1,25,000 / 2500 = 50 shares
    assert qty == 50


def test_fractional_kelly_fraction():
    sizer = PositionSizer(fractional_kelly_scalar=0.25)

    # Win rate 60% (0.60), Win/Loss ratio 2.0
    # Full Kelly: (0.60 * 2.0 - 0.40) / 2.0 = (1.20 - 0.40) / 2.0 = 0.80 / 2.0 = 0.40 (40%)
    # Quarter Kelly: 0.25 * 0.40 = 0.10 (10%)
    f_kelly = sizer.calculate_kelly_fraction(win_rate=0.60, win_loss_ratio=2.0)
    assert pytest.approx(f_kelly, 0.01) == 0.10
