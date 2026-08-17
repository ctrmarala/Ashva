"""
Ashva Institutional Backtesting Engine
Vectorized & Event-driven backtester with exact Indian regulatory taxes (STT, GST, Stamp Duty, SEBI),
Angel One brokerage, and slippage modeling.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.analytics.indian_costs import IndianCostModel, Segment, TradeCostBreakdown


@dataclass
class BacktestTrade:
    trade_id: int
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str                # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    cost_breakdown: TradeCostBreakdown
    duration_bars: int


@dataclass
class BacktestResult:
    symbol: str
    strategy_id: str
    initial_capital: float
    final_equity: float
    total_net_pnl: float
    net_roi_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    gross_profit_factor: float
    net_profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    total_brokerage_paid: float
    total_stt_paid: float
    total_taxes_paid: float
    equity_curve: pd.Series
    trade_list: List[BacktestTrade]

    def summary(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(self.final_equity, 2),
            "total_net_pnl": round(self.total_net_pnl, 2),
            "net_roi_pct": round(self.net_roi_pct, 2),
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate_pct, 2),
            "net_profit_factor": round(self.net_profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_brokerage": round(self.total_brokerage_paid, 2),
            "total_stt": round(self.total_stt_paid, 2),
            "total_taxes_and_charges": round(self.total_taxes_paid, 2),
        }


class BacktestEngine:
    """
    Simulates strategy execution and computes institutional performance metrics.
    """

    def __init__(
        self,
        cost_model: Optional[IndianCostModel] = None,
        initial_capital: float = 500000.0,  # ₹5,00,000 INR
        segment: Segment = Segment.EQUITY_INTRADAY,
    ):
        self.cost_model = cost_model or IndianCostModel()
        self.initial_capital = initial_capital
        self.segment = segment

    def run(
        self,
        df_with_signals: pd.DataFrame,
        symbol: str = "ASSET",
        strategy_id: str = "STRATEGY",
        capital_per_trade_pct: float = 0.95,
    ) -> BacktestResult:
        """
        Executes backtest over DataFrame containing 'close' and 'signal' (+1, -1, 0).
        """
        if "signal" not in df_with_signals.columns or "close" not in df_with_signals.columns:
            raise ValueError("DataFrame must contain 'close' and 'signal' columns")

        df = df_with_signals.copy()
        signals = df["signal"].values
        closes = df["close"].values
        indices = df.index

        trades: List[BacktestTrade] = []
        equity = self.initial_capital
        equity_series = [equity]
        equity_timestamps = [indices[0]]

        in_position = False
        position_side = None
        entry_idx = 0
        entry_price = 0.0
        trade_id = 1

        for i in range(len(df)):
            curr_signal = signals[i]
            curr_price = closes[i]

            # Entry Logic
            if not in_position and curr_signal != 0.0:
                in_position = True
                position_side = "LONG" if curr_signal > 0 else "SHORT"
                entry_idx = i
                entry_price = curr_price

            # Exit Logic
            elif in_position and (curr_signal == 0.0 or (curr_signal > 0 and position_side == "SHORT") or (curr_signal < 0 and position_side == "LONG")):
                exit_price = curr_price
                
                # Position Sizing (using available equity)
                allocated_capital = equity * capital_per_trade_pct
                quantity = max(1, int(allocated_capital / entry_price))

                if position_side == "LONG":
                    cost_breakdown = self.cost_model.calculate_trade_costs(
                        buy_price=entry_price,
                        sell_price=exit_price,
                        quantity=quantity,
                        segment=self.segment,
                    )
                else:  # SHORT
                    cost_breakdown = self.cost_model.calculate_trade_costs(
                        buy_price=exit_price,
                        sell_price=entry_price,
                        quantity=quantity,
                        segment=self.segment,
                    )

                trade = BacktestTrade(
                    trade_id=trade_id,
                    symbol=symbol,
                    entry_time=indices[entry_idx],
                    exit_time=indices[i],
                    side=position_side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    gross_pnl=cost_breakdown.gross_pnl,
                    net_pnl=cost_breakdown.net_pnl,
                    cost_breakdown=cost_breakdown,
                    duration_bars=i - entry_idx,
                )
                trades.append(trade)
                trade_id += 1

                # Update Equity
                equity += cost_breakdown.net_pnl
                equity_series.append(equity)
                equity_timestamps.append(indices[i])

                # Re-enter if signal switched directly from LONG to SHORT or vice-versa
                if curr_signal != 0.0:
                    position_side = "LONG" if curr_signal > 0 else "SHORT"
                    entry_idx = i
                    entry_price = curr_price
                else:
                    in_position = False
                    position_side = None

        # Build Equity Series
        if len(equity_series) == 1:
            eq_series = pd.Series([self.initial_capital] * len(df), index=indices)
        else:
            eq_series = pd.Series(equity_series[1:], index=equity_timestamps[1:])
            eq_series = eq_series.reindex(indices).ffill().fillna(self.initial_capital)

        # Performance Calculations
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.net_pnl > 0])
        losing_trades = len([t for t in trades if t.net_pnl <= 0])
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_wins = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
        gross_losses = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
        gross_pf = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 0.0)

        net_wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        net_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        net_pf = (net_wins / net_losses) if net_losses > 0 else (99.0 if net_wins > 0 else 0.0)

        total_brokerage = sum(t.cost_breakdown.brokerage for t in trades)
        total_stt = sum(t.cost_breakdown.stt for t in trades)
        total_taxes = sum(t.cost_breakdown.total_tax_and_charges for t in trades)
        total_net_pnl = equity - self.initial_capital
        roi_pct = (total_net_pnl / self.initial_capital) * 100.0

        # Drawdown
        peak = np.maximum.accumulate(eq_series.values)
        dd = (peak - eq_series.values) / peak * 100.0
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        # Sharpe & Sortino (Periodic returns)
        eq_returns = eq_series.pct_change().dropna().values
        if len(eq_returns) > 1 and np.std(eq_returns) > 1e-8:
            sharpe = float((np.mean(eq_returns) / np.std(eq_returns)) * np.sqrt(252 * 75))
            downside_returns = eq_returns[eq_returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-8
            sortino = float((np.mean(eq_returns) / downside_std) * np.sqrt(252 * 75)) if downside_std > 1e-8 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        return BacktestResult(
            symbol=symbol,
            strategy_id=strategy_id,
            initial_capital=self.initial_capital,
            final_equity=equity,
            total_net_pnl=total_net_pnl,
            net_roi_pct=roi_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            gross_profit_factor=gross_pf,
            net_profit_factor=net_pf,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            max_drawdown_duration_bars=0,
            total_brokerage_paid=total_brokerage,
            total_stt_paid=total_stt,
            total_taxes_paid=total_taxes,
            equity_curve=eq_series,
            trade_list=trades,
        )
