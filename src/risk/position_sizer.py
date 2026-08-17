"""
Ashva Position Sizing Engine
Implements Fixed Fractional Risk, ATR Volatility Parity, and Fractional Kelly Sizing for Indian Equities.
"""

from typing import Dict, Any, Optional
import numpy as np


class PositionSizer:
    """
    Computes precise quantity sizing respecting portfolio risk budgets and stop loss boundaries.
    """

    def __init__(
        self,
        default_risk_per_trade_pct: float = 0.75,  # Risk at most 0.75% of capital per trade
        max_capital_per_trade_pct: float = 25.0,   # At most 25% capital in a single stock
        fractional_kelly_scalar: float = 0.25,     # 1/4 Kelly safety multiplier
    ):
        self.default_risk_per_trade_pct = default_risk_per_trade_pct
        self.max_capital_per_trade_pct = max_capital_per_trade_pct
        self.fractional_kelly_scalar = fractional_kelly_scalar

    def calculate_fixed_risk_quantity(
        self,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_pct: Optional[float] = None,
    ) -> int:
        """
        Sizes quantity such that if stop-loss is hit, total dollar loss equals risk_pct of equity.
        
        Qty = (Equity * Risk%) / |Entry - StopLoss|
        """
        r_pct = risk_pct if risk_pct is not None else self.default_risk_per_trade_pct
        risk_amount_inr = equity * (r_pct / 100.0)

        price_risk_per_share = abs(entry_price - stop_loss_price)
        if price_risk_per_share <= 0:
            price_risk_per_share = entry_price * 0.01  # Fallback 1% risk distance

        raw_qty = risk_amount_inr / price_risk_per_share

        # Cap by maximum capital limit
        max_capital_inr = equity * (self.max_capital_per_trade_pct / 100.0)
        max_qty_capital = max_capital_inr / entry_price

        final_qty = max(1, int(min(raw_qty, max_qty_capital)))
        return final_qty

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        win_loss_ratio: float,
    ) -> float:
        """
        Calculates Fractional Kelly Criterion allocation fraction:
        f* = (p * b - q) / b
        where p = win rate, q = 1 - p, b = win/loss ratio (avg win / avg loss).
        """
        p = np.clip(win_rate, 0.01, 0.99)
        q = 1.0 - p
        b = max(win_loss_ratio, 0.1)

        full_kelly = (p * b - q) / b
        if full_kelly <= 0:
            return 0.0  # Zero bet if negative expectancy

        fractional_kelly = full_kelly * self.fractional_kelly_scalar
        return float(np.clip(fractional_kelly, 0.0, self.max_capital_per_trade_pct / 100.0))
