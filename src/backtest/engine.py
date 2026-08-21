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
from src.analytics.metrics import calculate_daily_mtm_sharpe, calculate_daily_mtm_sortino, calculate_bar_level_sharpe, calculate_max_drawdown_pct
from src.backtest.intrabar_simulator import IntrabarSimulator


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
    exit_reason: str = "SIGNAL"  # "SIGNAL", "STOP_LOSS", "TAKE_PROFIT", "EOD", "INTRABAR_DATA_UNAVAILABLE"
    entry_rationale: str = ""
    sizing_rationale: str = ""
    mfe_pct: float = 0.0         # Maximum Favorable Excursion (%)
    mae_pct: float = 0.0         # Maximum Adverse Excursion (%)
    slippage_bps: float = 3.0
    initial_sl: float = 0.0
    initial_tp: float = 0.0
    stop_dist: float = 0.0
    is_intrabar_qualified: bool = True


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
    sharpe_ratio: float          # Standardized Daily MTM Sharpe Ratio
    sortino_ratio: float         # Standardized Daily MTM Sortino Ratio
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    total_brokerage_paid: float
    total_stt_paid: float
    total_taxes_paid: float
    equity_curve: pd.Series
    trade_list: List[BacktestTrade]
    bar_sharpe_ratio: float = 0.0  # Frequency-aware diagnostic bar Sharpe

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
        use_1m_intrabar: bool = True,
        data_lake: Optional[Any] = None,
        max_volume_participation_pct: float = 0.10,  # Max 10% of execution candle volume
    ):
        self.cost_model = cost_model or IndianCostModel()
        self.initial_capital = initial_capital
        self.segment = segment
        self.use_1m_intrabar = use_1m_intrabar
        self.max_volume_participation_pct = max_volume_participation_pct
        self.intrabar_sim = IntrabarSimulator(data_lake=data_lake) if use_1m_intrabar else None

    def run(
        self,
        df_with_signals: pd.DataFrame,
        symbol: str = "ASSET",
        strategy_id: str = "STRATEGY",
        capital_per_trade_pct: float = 0.50,
        risk_per_trade_pct: Optional[float] = None,  # e.g., 0.005 for 0.50% account risk per trade
        trailing_mode: str = "NONE",  # "NONE", "BREAK_EVEN", "STEP_RATCHET"
    ) -> BacktestResult:
        """
        Executes backtest over DataFrame containing 'close', 'signal' (+1, -1, 0),
        and optionally 'open', 'high', 'low', 'stop_loss', 'take_profit', 'rationale'.
        Supports Capital Sizing, Risk Budgeting, and Tiered Step-Ratchet Trailing Stops.
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

        has_rationales = "rationale" in df.columns or "entry_rationale" in df.columns
        rationale_col = "rationale" if "rationale" in df.columns else ("entry_rationale" if "entry_rationale" in df.columns else None)

        trades: List[BacktestTrade] = []
        cash = self.initial_capital
        bar_equity = np.full(n_bars, self.initial_capital, dtype=np.float64)

        in_position = False
        position_side = None
        entry_idx = 0
        entry_price = 0.0
        entry_qty = 0
        initial_sl = 0.0
        current_sl = 0.0
        current_tp = 0.0
        current_entry_rationale = ""
        trade_id = 1

        for i in range(n_bars - 1):
            curr_signal = signals[i]
            next_open = opens[i + 1]
            next_high = highs[i + 1]
            next_low = lows[i + 1]
            next_time = indices[i + 1]

            # 1. Evaluate Intrabar Stop Loss / Take Profit on active position
            if in_position:
                # Dynamic Step-Ratchet / Break-Even Trailing update
                if trailing_mode in ("STEP_RATCHET", "BREAK_EVEN") and initial_sl > 0:
                    if position_side == "LONG":
                        unit_risk = max(entry_price * 0.002, entry_price - initial_sl)
                        peak_high = np.max(highs[entry_idx : i + 1])
                        if trailing_mode == "STEP_RATCHET":
                            if peak_high >= entry_price + (2.0 * unit_risk):
                                current_sl = max(current_sl, entry_price + (1.50 * unit_risk))
                            elif peak_high >= entry_price + (1.5 * unit_risk):
                                current_sl = max(current_sl, entry_price + (0.75 * unit_risk))
                            elif peak_high >= entry_price + (1.0 * unit_risk):
                                current_sl = max(current_sl, entry_price + (0.05 * unit_risk))
                        elif trailing_mode == "BREAK_EVEN":
                            if peak_high >= entry_price + (1.0 * unit_risk):
                                current_sl = max(current_sl, entry_price + (0.05 * unit_risk))
                    else:  # SHORT
                        unit_risk = max(entry_price * 0.002, initial_sl - entry_price)
                        lowest_low = np.min(lows[entry_idx : i + 1])
                        if trailing_mode == "STEP_RATCHET":
                            if lowest_low <= entry_price - (2.0 * unit_risk):
                                current_sl = min(current_sl, entry_price - (1.50 * unit_risk))
                            elif lowest_low <= entry_price - (1.5 * unit_risk):
                                current_sl = min(current_sl, entry_price - (0.75 * unit_risk))
                            elif lowest_low <= entry_price - (1.0 * unit_risk):
                                current_sl = min(current_sl, entry_price - (0.05 * unit_risk))
                        elif trailing_mode == "BREAK_EVEN":
                            if lowest_low <= entry_price - (1.0 * unit_risk):
                                current_sl = min(current_sl, entry_price - (0.05 * unit_risk))

                exited_intrabar = False
                exit_price = 0.0
                exit_reason = "SIGNAL"
                mfe, mae = 0.0, 0.0

                # 1. High-Resolution 1-Minute Historical Intrabar Execution
                if self.intrabar_sim is not None and symbol != "ASSET":
                    max_eod_time = indices[entry_idx].replace(hour=15, minute=15, second=0) if self.segment == Segment.EQUITY_INTRADAY else None
                    sim_res = self.intrabar_sim.simulate_trade(
                        symbol=symbol,
                        entry_time=indices[entry_idx],
                        entry_price=entry_price,
                        side=position_side,
                        stop_loss=initial_sl,
                        take_profit=current_tp,
                        trailing_mode=trailing_mode,
                        max_exit_time=max_eod_time,
                    )
                    if sim_res.is_intrabar_qualified and (sim_res.exit_time <= next_time or i == (n_bars - 2) or (self.segment == Segment.EQUITY_INTRADAY and sim_res.exit_time.date() == indices[entry_idx].date())):
                        exited_intrabar = True
                        exit_price = sim_res.exit_price
                        exit_reason = sim_res.exit_reason
                        next_time = sim_res.exit_time
                        mfe, mae = sim_res.mfe_pct, sim_res.mae_pct

                # 2. Native Multi-Timeframe Intrabar Fallback
                if not exited_intrabar and not (self.segment == Segment.EQUITY_INTRADAY and indices[i].date() != next_time.date()):
                    if position_side == "LONG":
                        if current_sl > 0 and next_low <= current_sl:
                            exited_intrabar = True
                            exit_price = min(next_open, current_sl)
                            exit_reason = "TRAILING_STOP_RATCHET" if current_sl > initial_sl else "STOP_LOSS"
                        elif current_tp > 0 and next_high >= current_tp:
                            exited_intrabar = True
                            exit_price = max(next_open, current_tp)
                            exit_reason = "TAKE_PROFIT"
                    else:  # SHORT
                        if current_sl > 0 and next_high >= current_sl:
                            exited_intrabar = True
                            exit_price = max(next_open, current_sl)
                            exit_reason = "TRAILING_STOP_RATCHET" if current_sl < initial_sl else "STOP_LOSS"
                        elif current_tp > 0 and next_low <= current_tp:
                            exited_intrabar = True
                            exit_price = min(next_open, current_tp)
                            exit_reason = "TAKE_PROFIT"

                if exited_intrabar:
                    # Strict SEBI Overnight Short Prohibition Guard
                    if position_side == "SHORT" and indices[entry_idx].date() != next_time.date():
                        raise ValueError(f"CRITICAL SEBI VIOLATION: Cash equity short position in {symbol} spanned across multiple dates ({indices[entry_idx].date()} to {next_time.date()}).")

                    is_sl = ("STOP_LOSS" in exit_reason or "TRAILING" in exit_reason)
                    cost_breakdown = self.cost_model.calculate_trade_costs(
                        buy_price=entry_price if position_side == "LONG" else exit_price,
                        sell_price=exit_price if position_side == "LONG" else entry_price,
                        quantity=entry_qty,
                        segment=self.segment,
                        is_stop_loss=is_sl,
                    )
                    cash += cost_breakdown.net_pnl
                    bar_equity[i + 1] = cash

                    # Compute MFE and MAE if not populated by simulator
                    if mfe == 0.0 and mae == 0.0:
                        trade_highs = highs[entry_idx : i + 2]
                        trade_lows = lows[entry_idx : i + 2]
                        if len(trade_highs) > 0 and entry_price > 0:
                            if position_side == "LONG":
                                mfe = ((np.max(trade_highs) - entry_price) / entry_price) * 100.0
                                mae = ((np.min(trade_lows) - entry_price) / entry_price) * 100.0
                            else:
                                mfe = ((entry_price - np.min(trade_lows)) / entry_price) * 100.0
                                mae = ((entry_price - np.max(trade_highs)) / entry_price) * 100.0

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
                            entry_rationale=f"{strategy_id} {position_side} Trigger @ {indices[entry_idx]}",
                            sizing_rationale=f"Qty {entry_qty} (Stop Dist: Rs {abs(entry_price - initial_sl):.2f})",
                            mfe_pct=round(mfe, 2),
                            mae_pct=round(mae, 2),
                            slippage_bps=self.cost_model.default_slippage_bps,
                            initial_sl=initial_sl,
                            initial_tp=current_tp,
                            stop_dist=round(abs(entry_price - initial_sl), 2) if initial_sl > 0 else 0.0,
                            is_intrabar_qualified=(exit_reason != "INTRABAR_DATA_UNAVAILABLE"),
                        )
                    )
                    trade_id += 1
                    in_position = False
                    position_side = None

            if self.segment == Segment.EQUITY_INTRADAY and indices[i].date() != next_time.date():
                curr_signal = 0.0

            # 2. Evaluate Signal Changes for Next-Bar Open Execution
            if not in_position and curr_signal != 0.0:
                in_position = True
                position_side = "LONG" if curr_signal > 0 else "SHORT"
                entry_idx = i + 1
                entry_price = next_open
                current_sl = stop_losses[i] if has_stops else 0.0
                initial_sl = current_sl
                current_tp = take_profits[i] if has_stops else 0.0

                if has_rationales and rationale_col is not None:
                    rat_val = str(df[rationale_col].iloc[i])
                    current_entry_rationale = rat_val if rat_val and rat_val != "nan" else f"{strategy_id} {position_side} Trigger @ {indices[entry_idx]}"
                else:
                    current_entry_rationale = f"{strategy_id} {position_side} Trigger @ {indices[entry_idx]}"

                # Risk-Based Sizing vs Capital-Percentage Sizing
                if risk_per_trade_pct is not None and current_sl > 0:
                    stop_dist = abs(entry_price - current_sl)
                    if stop_dist > 0.05:
                        risk_amt = cash * risk_per_trade_pct
                        risk_qty = int(risk_amt / stop_dist)
                        max_cap_qty = int((cash * capital_per_trade_pct) / entry_price)
                        entry_qty = min(risk_qty, max_cap_qty)
                    else:
                        entry_qty = int((cash * capital_per_trade_pct) / entry_price)
                else:
                    allocated_capital = cash * capital_per_trade_pct
                    entry_qty = int(allocated_capital / entry_price)

                # Liquidity Capacity Participation Cap (Max 10% of execution bar volume)
                if "volume" in df.columns:
                    bar_vol = float(df["volume"].iloc[entry_idx])
                    max_liq_qty = max(1, int(bar_vol * self.max_volume_participation_pct))
                    entry_qty = min(entry_qty, max_liq_qty)

                # Institutional zero-risk budget policy: If risk size < 1 share -> NO TRADE
                if entry_qty < 1:
                    in_position = False
                    position_side = None
                    continue

            elif in_position and (
                curr_signal == 0.0
                or (curr_signal > 0 and position_side == "SHORT")
                or (curr_signal < 0 and position_side == "LONG")
            ):
                if self.segment == Segment.EQUITY_INTRADAY and indices[entry_idx].date() != next_time.date():
                    exit_price = closes[i]
                    next_time = indices[i]
                    exit_reason_str = "EOD_SQUARE_OFF"
                else:
                    exit_price = next_open
                    exit_reason_str = "SIGNAL"

                cost_breakdown = self.cost_model.calculate_trade_costs(
                    buy_price=entry_price if position_side == "LONG" else exit_price,
                    sell_price=exit_price if position_side == "LONG" else entry_price,
                    quantity=entry_qty,
                    segment=self.segment,
                    is_stop_loss=False,
                )
                cash += cost_breakdown.net_pnl
                bar_equity[i + 1] = cash

                # Compute MFE and MAE
                trade_highs = highs[entry_idx : i + 2]
                trade_lows = lows[entry_idx : i + 2]
                if len(trade_highs) > 0 and entry_price > 0:
                    if position_side == "LONG":
                        mfe = ((np.max(trade_highs) - entry_price) / entry_price) * 100.0
                        mae = ((np.min(trade_lows) - entry_price) / entry_price) * 100.0
                    else:
                        mfe = ((entry_price - np.min(trade_lows)) / entry_price) * 100.0
                        mae = ((entry_price - np.max(trade_highs)) / entry_price) * 100.0
                else:
                    mfe, mae = 0.0, 0.0

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
                        exit_reason=exit_reason_str,
                        entry_rationale=current_entry_rationale or f"{strategy_id} {position_side} Trigger @ {indices[entry_idx]}",
                        sizing_rationale=f"Qty {entry_qty} (Stop Dist: Rs {abs(entry_price - initial_sl):.2f})",
                        mfe_pct=round(mfe, 2),
                        mae_pct=round(mae, 2),
                        slippage_bps=self.cost_model.default_slippage_bps,
                        initial_sl=initial_sl,
                        initial_tp=current_tp,
                        stop_dist=round(abs(entry_price - initial_sl), 2) if initial_sl > 0 else 0.0,
                        is_intrabar_qualified=True,
                    )
                )
                trade_id += 1

                if curr_signal != 0.0:
                    position_side = "LONG" if curr_signal > 0 else "SHORT"
                    entry_idx = i + 1
                    entry_price = next_open
                    current_sl = stop_losses[i] if has_stops else 0.0
                    initial_sl = current_sl
                    current_tp = take_profits[i] if has_stops else 0.0

                    if has_rationales and rationale_col is not None:
                        rat_val = str(df[rationale_col].iloc[i])
                        current_entry_rationale = rat_val if rat_val and rat_val != "nan" else f"{strategy_id} {position_side} Trigger @ {indices[entry_idx]}"
                    else:
                        current_entry_rationale = f"{strategy_id} {position_side} Trigger @ {indices[entry_idx]}"

                    if risk_per_trade_pct is not None and current_sl > 0:
                        stop_dist = abs(entry_price - current_sl)
                        if stop_dist > 0.05:
                            risk_amt = cash * risk_per_trade_pct
                            risk_qty = int(risk_amt / stop_dist)
                            max_cap_qty = int((cash * capital_per_trade_pct) / entry_price)
                            entry_qty = max(1, min(risk_qty, max_cap_qty))
                        else:
                            entry_qty = max(1, int((cash * capital_per_trade_pct) / entry_price))
                    else:
                        allocated_capital = cash * capital_per_trade_pct
                        entry_qty = max(1, int(allocated_capital / entry_price))

                    # Liquidity Capacity Participation Cap (Max 10% of execution bar volume)
                    if "volume" in df.columns:
                        bar_vol = float(df["volume"].iloc[entry_idx])
                        max_liq_qty = max(1, int(bar_vol * self.max_volume_participation_pct))
                        entry_qty = min(entry_qty, max_liq_qty)

                    if entry_qty < 1:
                        in_position = False
                        position_side = None
                        continue
                else:
                    in_position = False
                    position_side = None
                    current_entry_rationale = ""

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

            trade_highs = highs[entry_idx : last_idx + 1]
            trade_lows = lows[entry_idx : last_idx + 1]
            if len(trade_highs) > 0 and entry_price > 0:
                if position_side == "LONG":
                    mfe = ((np.max(trade_highs) - entry_price) / entry_price) * 100.0
                    mae = ((np.min(trade_lows) - entry_price) / entry_price) * 100.0
                else:
                    mfe = ((entry_price - np.min(trade_lows)) / entry_price) * 100.0
                    mae = ((entry_price - np.max(trade_highs)) / entry_price) * 100.0
            else:
                mfe, mae = 0.0, 0.0

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
                    entry_rationale=f"{strategy_id} {position_side} Trigger @ {indices[entry_idx]}",
                    sizing_rationale=f"Qty {entry_qty}",
                    mfe_pct=round(mfe, 2),
                    mae_pct=round(mae, 2),
                    slippage_bps=self.cost_model.default_slippage_bps,
                    initial_sl=initial_sl,
                    initial_tp=current_tp,
                    stop_dist=round(abs(entry_price - initial_sl), 2) if initial_sl > 0 else 0.0,
                    is_intrabar_qualified=True,
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

        # Standardized Daily MTM Sharpe and Sortino Ratios
        daily_sharpe = calculate_daily_mtm_sharpe(equity_df)
        daily_sortino = calculate_daily_mtm_sortino(equity_df)
        max_drawdown = calculate_max_drawdown_pct(equity_df)

        returns = equity_df.pct_change().dropna()
        bar_sharpe = calculate_bar_level_sharpe(returns.values, bars_per_day=25.0)

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
            sharpe_ratio=round(daily_sharpe, 3),
            sortino_ratio=round(daily_sortino, 3),
            max_drawdown_pct=round(max_drawdown, 2),
            max_drawdown_duration_bars=0,
            total_brokerage_paid=total_brokerage,
            total_stt_paid=total_stt,
            total_taxes_paid=total_taxes,
            equity_curve=equity_df,
            trade_list=trades,
            bar_sharpe_ratio=round(bar_sharpe, 3),
        )

    def run_slippage_stress_matrix(
        self,
        df_with_signals: pd.DataFrame,
        symbol: str = "ASSET",
        strategy_id: str = "STRATEGY",
        capital_per_trade_pct: float = 0.95,
    ) -> pd.DataFrame:
        """
        Executes a 5-tier institutional slippage sensitivity stress test:
        - Optimistic:   1 bps (0.01%)
        - Base:         3 bps (0.03%)
        - Conservative: 5 bps (0.05%)
        - Stress:       10 bps (0.10%)
        - Extreme:      20 bps (0.20%)
        """
        tiers = [
            ("Optimistic", 1.0),
            ("Base", 3.0),
            ("Conservative", 5.0),
            ("Stress", 10.0),
            ("Extreme", 20.0),
        ]
        results = []
        for name, bps in tiers:
            cost_model = IndianCostModel(slippage_bps=bps)
            eng = BacktestEngine(cost_model=cost_model, initial_capital=self.initial_capital, segment=self.segment)
            res = eng.run(df_with_signals, symbol=symbol, strategy_id=strategy_id, capital_per_trade_pct=capital_per_trade_pct)
            results.append({
                "Scenario": name,
                "Slippage_Bps": bps,
                "Net_Pnl_INR": round(res.total_net_pnl, 2),
                "Net_ROI_Pct": round(res.net_roi_pct, 2),
                "Profit_Factor": round(res.net_profit_factor, 2),
                "Sharpe": round(res.sharpe_ratio, 2),
                "MaxDD_Pct": round(res.max_drawdown_pct, 2),
                "Total_Taxes_INR": round(res.total_taxes_paid, 2),
            })
        return pd.DataFrame(results)
