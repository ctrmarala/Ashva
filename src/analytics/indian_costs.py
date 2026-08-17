"""
Ashva Indian Market Regulatory Cost & Friction Engine
Implements exact statutory taxes, exchange fees, SEBI levies, Angel One brokerage,
and market slippage models for Indian Equities and Derivatives (NSE/BSE).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class Segment(str, Enum):
    EQUITY_INTRADAY = "EQUITY_INTRADAY"   # MIS
    EQUITY_DELIVERY = "EQUITY_DELIVERY"   # CNC
    FUTURES = "FUTURES"                   # NFO Futures
    OPTIONS = "OPTIONS"                   # NFO Options


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"


@dataclass(frozen=True)
class TradeCostBreakdown:
    buy_turnover: float
    sell_turnover: float
    total_turnover: float
    gross_pnl: float
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    slippage_cost: float
    total_tax_and_charges: float
    net_pnl: float
    net_return_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buy_turnover": round(self.buy_turnover, 2),
            "sell_turnover": round(self.sell_turnover, 2),
            "total_turnover": round(self.total_turnover, 2),
            "gross_pnl": round(self.gross_pnl, 2),
            "brokerage": round(self.brokerage, 2),
            "stt": round(self.stt, 2),
            "exchange_charges": round(self.exchange_charges, 2),
            "gst": round(self.gst, 2),
            "sebi_charges": round(self.sebi_charges, 2),
            "stamp_duty": round(self.stamp_duty, 2),
            "slippage_cost": round(self.slippage_cost, 2),
            "total_tax_and_charges": round(self.total_tax_and_charges, 2),
            "net_pnl": round(self.net_pnl, 2),
            "net_return_pct": round(self.net_return_pct, 4),
        }


class IndianCostModel:
    """
    Computes precise statutory and execution costs for Indian financial markets.
    Rates comply with SEBI, Ministry of Finance, and Angel One pricing schedules.
    """

    # Exchange Turnover Rates
    EXCHANGE_TURNOVER_RATES = {
        Exchange.NSE: 0.0000325,  # 0.00325%
        Exchange.BSE: 0.0000375,  # 0.00375%
    }

    # SEBI Charges (Rs 10 per crore)
    SEBI_RATE = 0.0000010  # 0.00010%

    # GST Rate (18% on Brokerage + Exchange Turnover + SEBI)
    GST_RATE = 0.18

    # STT Rates (Securities Transaction Tax)
    STT_RATES = {
        Segment.EQUITY_INTRADAY: {"buy": 0.0, "sell": 0.00025},       # 0.025% on sell
        Segment.EQUITY_DELIVERY: {"buy": 0.001, "sell": 0.001},       # 0.1% on buy & sell
        Segment.FUTURES: {"buy": 0.0, "sell": 0.000125},              # 0.0125% on sell
        Segment.OPTIONS: {"buy": 0.0, "sell": 0.000625},              # 0.0625% on sell (premium)
    }

    # Stamp Duty Rates (levied only on the BUY side since 1 July 2020)
    STAMP_DUTY_RATES = {
        Segment.EQUITY_INTRADAY: 0.00003,   # 0.003% on buy
        Segment.EQUITY_DELIVERY: 0.00015,   # 0.015% on buy
        Segment.FUTURES: 0.00002,           # 0.002% on buy
        Segment.OPTIONS: 0.00003,           # 0.003% on buy
    }

    def __init__(
        self,
        brokerage_per_order: float = 20.0,
        brokerage_pct_cap: float = 0.0003,  # 0.03% cap
        exchange: Exchange = Exchange.NSE,
        default_slippage_bps: float = 3.0,  # 3 basis points = 0.03%
    ):
        self.brokerage_per_order = brokerage_per_order
        self.brokerage_pct_cap = brokerage_pct_cap
        self.exchange = exchange
        self.default_slippage_bps = default_slippage_bps

    def calculate_brokerage(self, turnover: float) -> float:
        """
        Angel One standard rate: Flat Rs 20 or 0.03% (whichever is lower).
        """
        variable_brokerage = turnover * self.brokerage_pct_cap
        return min(self.brokerage_per_order, variable_brokerage)

    def calculate_trade_costs(
        self,
        buy_price: float,
        sell_price: float,
        quantity: int,
        segment: Segment = Segment.EQUITY_INTRADAY,
        slippage_bps: float = None,
    ) -> TradeCostBreakdown:
        """
        Calculates exact net PnL and fee breakdown for a complete roundtrip trade.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if buy_price <= 0 or sell_price <= 0:
            raise ValueError("Prices must be positive numbers")

        if slippage_bps is None:
            slippage_bps = self.default_slippage_bps

        buy_turnover = buy_price * quantity
        sell_turnover = sell_price * quantity
        total_turnover = buy_turnover + sell_turnover
        gross_pnl = (sell_price - buy_price) * quantity

        # 1. Brokerage (Buy Leg + Sell Leg)
        buy_brokerage = self.calculate_brokerage(buy_turnover)
        sell_brokerage = self.calculate_brokerage(sell_turnover)
        total_brokerage = buy_brokerage + sell_brokerage

        # 2. STT (Securities Transaction Tax)
        stt_buy = buy_turnover * self.STT_RATES[segment]["buy"]
        stt_sell = sell_turnover * self.STT_RATES[segment]["sell"]
        total_stt = round(stt_buy + stt_sell)  # STT is rounded to nearest integer by exchanges

        # 3. Exchange Turnover Charges
        exch_rate = self.EXCHANGE_TURNOVER_RATES.get(self.exchange, 0.0000325)
        exchange_charges = total_turnover * exch_rate

        # 4. SEBI Charges
        sebi_charges = total_turnover * self.SEBI_RATE

        # 5. GST (18% on Brokerage + Exchange + SEBI)
        gst = (total_brokerage + exchange_charges + sebi_charges) * self.GST_RATE

        # 6. Stamp Duty (Buy turnover only)
        stamp_rate = self.STAMP_DUTY_RATES[segment]
        stamp_duty = round(buy_turnover * stamp_rate)

        # 7. Slippage Cost
        slippage_factor = slippage_bps / 10000.0
        slippage_cost = total_turnover * slippage_factor

        # Total Deductions
        total_tax_and_charges = (
            total_brokerage
            + total_stt
            + exchange_charges
            + gst
            + sebi_charges
            + stamp_duty
            + slippage_cost
        )

        net_pnl = gross_pnl - total_tax_and_charges
        net_return_pct = (net_pnl / buy_turnover) * 100.0 if buy_turnover > 0 else 0.0

        return TradeCostBreakdown(
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            total_turnover=total_turnover,
            gross_pnl=gross_pnl,
            brokerage=total_brokerage,
            stt=float(total_stt),
            exchange_charges=exchange_charges,
            gst=gst,
            sebi_charges=sebi_charges,
            stamp_duty=float(stamp_duty),
            slippage_cost=slippage_cost,
            total_tax_and_charges=total_tax_and_charges,
            net_pnl=net_pnl,
            net_return_pct=net_return_pct,
        )
