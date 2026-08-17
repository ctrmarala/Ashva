"""
Ashva Institutional Backtesting Engine
High-fidelity, realistic backtester with:
1. Strict Next-Bar Open Execution (Signal at Bar t Close -> Fill at Bar t+1 Open) to eliminate lookahead bias.
2. Decision-Time Position Sizing (Quantity computed strictly at Entry).
3. Intrabar Stop Loss and Take Profit evaluation against Bar High/Low.
4. Continuous Bar-by-Bar Mark-to-Market (MTM) Portfolio Equity Curve for accurate continuous Sharpe, Sortino & MaxDD.
5. Exact Indian statutory taxes (STT, GST, Stamp Duty, SEBI turnover) and Angel One brokerage modeling.
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
    exit_reason: str = "SIGNAL"  # "SIGNAL", "STOP_LOSS", "TAKE_PROFIT", "EOD"


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
    Realistic quantitative execution backtester with next-bar execution conventions and continuous MTM equity tracking.
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
        Executes backtest over DataFrame containing 'close', 'signal' (+1, -1, 0),
        and optionally 'open', 'high', 'low', 'stop_loss', 'take_profit'.
        """
        if "signal" not in df_with_signals.columns or "close" not in df_with_signals.columns:
            raise ValueError("DataFrame must contain 'close' and 'signal' columns")

        df = df_with_signals.copy()
        signals = df["signal"].values
        closes = df["close"].values
        opens = df["open"].values if "open" in df.columns else closes
        highs = df["high"].values if "high" in df.columns else closes
        lows = df["low"].values if "low" in df.columns else closes
        indices = df.index
        n_bars = len(df)

        has_stops = "stop_loss" in df.columns and "take_profit" in df.columns
        stop_losses = df["stop_loss"].values if has_stops else np.zeros(n_bars)
        take_profits = df["take_profit"].values if has_stops else np.zeros(n_bars)

        trades: List[BacktestTrade] = []
        cash = self.initial_capital
        bar_equity = np.full(n_bars, self.initial_capital, dtype=np.float64)

        in_position = False
        position_side = None
        entry_idx = 0
        entry_price = 0.0
        entry_qty = 0
        current_sl = 0.0
        current_tp = 0.0
        trade_id = 1

        for i in range(n_bars - 1):
            curr_signal = signals[i]
            next_open = opens[i + 1]
            next_high = highs[i + 1]
            next_low = lows[i + 1]
            next_time = indices[i + 1]

            # 1. Evaluate Intrabar Stop Loss / Take Profit on active position
            if in_position:
                exited_intrabar = False
                exit_price = 0.0
                exit_reason = "SIGNAL"

                if position_side == "LONG":
                    if current_sl > 0 and next_low <= current_sl:
                        exited_intrabar = True
                        exit_price = min(next_open, current_sl)
                        exit_reason = "STOP_LOSS"
                    elif current_tp > 0 and next_high >= current_tp:
                        exited_intrabar = True
                        exit_price = max(next_open, current_tp)
                        exit_reason = "TAKE_PROFIT"
                else:  # SHORT
                    if current_sl > 0 and next_high >= current_sl:
                        exited_intrabar = True
                        exit_price = max(next_open, current_sl)
                        exit_reason = "STOP_LOSS"
                    elif current_tp > 0 and next_low <= current_tp:
                        exited_intrabar = True
                        exit_price = min(next_open, current_tp)
                        exit_reason = "TAKE_PROFIT"

                if exited_intrabar:
                    cost_breakdown = self.cost_model.calculate_trade_costs(
                        buy_price=entry_price if position_side == "LONG" else exit_price,
                        sell_price=exit_price if position_side == "LONG" else entry_price,
                        quantity=entry_qty,
                        segment=self.segment,
                    )
                    cash += cost_breakdown.net_pnl
                    bar_equity[i + 1] = cash

                    trades.append(
                        BacktestTrade(
                            trade_id=trade_id,
                            symbol=symbol,
                            entry_time=indices[entry_idx],
                            exit_time=next_time,
                            side=position_side,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=entry_qty,
                            gross_pnl=cost_breakdown.gross_pnl,
                            net_pnl=cost_breakdown.net_pnl,
                            cost_breakdown=cost_breakdown,
                            duration_bars=(i + 1 - entry_idx),
                            exit_reason=exit_reason,
                        )
                    )
                    trade_id += 1
                    in_position = False
                    position_side = None
                    continue

            # 2. Evaluate Signal Changes for Next-Bar Open Execution
            if not in_position and curr_signal != 0.0:
                in_position = True
                position_side = "LONG" if curr_signal > 0 else "SHORT"
                entry_idx = i + 1
                entry_price = next_open
                current_sl = stop_losses[i] if has_stops else 0.0
                current_tp = take_profits[i] if has_stops else 0.0

                allocated_capital = cash * capital_per_trade_pct
                entry_qty = max(1, int(allocated_capital / entry_price))

            elif in_position and (
                curr_signal == 0.0
                or (curr_signal > 0 and position_side == "SHORT")
                or (curr_signal < 0 and position_side == "LONG")
            ):
                exit_price = next_open
                cost_breakdown = self.cost_model.calculate_trade_costs(
                    buy_price=entry_price if position_side == "LONG" else exit_price,
                    sell_price=exit_price if position_side == "LONG" else entry_price,
                    quantity=entry_qty,
                    segment=self.segment,
                )
                cash += cost_breakdown.net_pnl
                bar_equity[i + 1] = cash

                trades.append(
                    BacktestTrade(
                        trade_id=trade_id,
                        symbol=symbol,
                        entry_time=indices[entry_idx],
                        exit_time=next_time,
                        side=position_side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=entry_qty,
                        gross_pnl=cost_breakdown.gross_pnl,
                        net_pnl=cost_breakdown.net_pnl,
                        cost_breakdown=cost_breakdown,
                        duration_bars=(i + 1 - entry_idx),
                        exit_reason="SIGNAL",
                    )
                )
                trade_id += 1

                if curr_signal != 0.0:
                    position_side = "LONG" if curr_signal > 0 else "SHORT"
                    entry_idx = i + 1
                    entry_price = next_open
                    current_sl = stop_losses[i] if has_stops else 0.0
                    current_tp = take_profits[i] if has_stops else 0.0
                    allocated_capital = cash * capital_per_trade_pct
                    entry_qty = max(1, int(allocated_capital / entry_price))
                else:
                    in_position = False
                    position_side = None

            # 3. Continuous Mark-to-Market Bar Equity Update
            if in_position:
                unrealized_mtm = entry_qty * (closes[i + 1] - entry_price) * (1.0 if position_side == "LONG" else -1.0)
                bar_equity[i + 1] = cash + unrealized_mtm
            else:
                bar_equity[i + 1] = cash

        # Close any remaining position on the final bar
        if in_position:
            last_idx = n_bars - 1
            exit_price = closes[last_idx]
            cost_breakdown = self.cost_model.calculate_trade_costs(
                buy_price=entry_price if position_side == "LONG" else exit_price,
                sell_price=exit_price if position_side == "LONG" else entry_price,
                quantity=entry_qty,
                segment=self.segment,
            )
            cash += cost_breakdown.net_pnl
            bar_equity[last_idx] = cash

            trades.append(
                BacktestTrade(
                    trade_id=trade_id,
                    symbol=symbol,
                    entry_time=indices[entry_idx],
                    exit_time=indices[last_idx],
                    side=position_side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=entry_qty,
                    gross_pnl=cost_breakdown.gross_pnl,
                    net_pnl=cost_breakdown.net_pnl,
                    cost_breakdown=cost_breakdown,
                    duration_bars=(last_idx - entry_idx),
                    exit_reason="EOD",
                )
            )

        # Compute Continuous Quant Statistics
        equity_df = pd.Series(bar_equity, index=indices)
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.net_pnl > 0])
        losing_trades = len([t for t in trades if t.net_pnl <= 0])
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        final_equity = float(bar_equity[-1])
        total_net_pnl = final_equity - self.initial_capital
        net_roi = (total_net_pnl / self.initial_capital) * 100.0

        gross_wins = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
        gross_losses = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
        gross_profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (99.0 if gross_wins > 0 else 0.0)

        net_wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        net_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        net_profit_factor = (net_wins / net_losses) if net_losses > 0 else (99.0 if net_wins > 0 else 0.0)

        # Continuous Mark-to-Market Drawdown & Returns
        cum_max = equity_df.cummax()
        drawdowns = (equity_df - cum_max) / cum_max
        max_drawdown = abs(drawdowns.min()) * 100.0 if not drawdowns.empty else 0.0

        returns = equity_df.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252 * 25))
            downside = returns[returns < 0]
            sortino = float((returns.mean() / downside.std()) * np.sqrt(252 * 25)) if len(downside) > 0 and downside.std() > 0 else sharpe
        else:
            sharpe = 0.0
            sortino = 0.0

        total_brokerage = sum(t.cost_breakdown.brokerage for t in trades)
        total_stt = sum(t.cost_breakdown.stt for t in trades)
        total_taxes = sum(t.cost_breakdown.total_tax_and_charges for t in trades)

        return BacktestResult(
            symbol=symbol,
            strategy_id=strategy_id,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_net_pnl=total_net_pnl,
            net_roi_pct=net_roi,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            gross_profit_factor=gross_profit_factor,
            net_profit_factor=net_profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_drawdown,
            max_drawdown_duration_bars=0,
            total_brokerage_paid=total_brokerage,
            total_stt_paid=total_stt,
            total_taxes_paid=total_taxes,
            equity_curve=equity_df,
            trade_list=trades,
        )
