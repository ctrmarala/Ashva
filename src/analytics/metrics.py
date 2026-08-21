"""
Ashva Canonical Quantitative Performance Metrics Engine
Standardizes:
1. Daily Mark-to-Market (MTM) Annualized Sharpe and Sortino Ratios.
2. Continuous Equity Curve Maximum Drawdown, Calmar, and Recovery Factor.
3. Frequency-aware Bar-Level Sharpe diagnostics.
4. Discrete Trade-Level Expectancy, Profit Factor, Win Rate, Payoff Ratio, MFE/MAE.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd


def calculate_daily_mtm_sharpe(
    equity_series: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """
    Canonical Daily Mark-to-Market Sharpe Ratio.
    Computes daily returns from a continuous chronological daily equity curve:
        Sharpe = mean(daily excess return) / std(daily excess return) * sqrt(252)
    """
    if equity_series.empty or len(equity_series) < 3:
        return 0.0

    # Resample or group by calendar date taking the last daily MTM equity
    if isinstance(equity_series.index, pd.DatetimeIndex):
        daily_equity = equity_series.resample("D").last().dropna()
    else:
        daily_equity = equity_series.dropna()

    if len(daily_equity) < 3:
        return 0.0

    daily_returns = daily_equity.pct_change().dropna()
    clean_rets = daily_returns.values

    # Remove zero-variance / non-finite entries
    clean_rets = clean_rets[np.isfinite(clean_rets)]
    if len(clean_rets) < 2:
        return 0.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0 if risk_free_rate > 0 else 0.0
    excess_rets = clean_rets - daily_rf
    mean_excess = np.mean(excess_rets)
    std_dev = np.std(excess_rets, ddof=1)
    if std_dev < 1e-8:
        return 99.0 if mean_excess > 0 else 0.0

    return float((mean_excess / std_dev) * np.sqrt(annualization_factor))


def calculate_daily_mtm_sortino(
    equity_series: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """
    Canonical Daily Mark-to-Market Sortino Ratio.
    Penalizes only downside volatility below target/risk-free rate.
    """
    if equity_series.empty or len(equity_series) < 3:
        return 0.0

    if isinstance(equity_series.index, pd.DatetimeIndex):
        daily_equity = equity_series.resample("D").last().dropna()
    else:
        daily_equity = equity_series.dropna()

    if len(daily_equity) < 3:
        return 0.0

    daily_returns = daily_equity.pct_change().dropna()
    clean_rets = daily_returns.values
    clean_rets = clean_rets[np.isfinite(clean_rets)]
    if len(clean_rets) < 2:
        return 0.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0 if risk_free_rate > 0 else 0.0
    excess_rets = clean_rets - daily_rf

    downside_rets = excess_rets[excess_rets < 0]
    if len(downside_rets) < 2:
        # If no downside, fallback to regular Sharpe or high value
        return calculate_daily_mtm_sharpe(equity_series, risk_free_rate, annualization_factor)

    downside_std = np.std(downside_rets, ddof=1)
    if downside_std < 1e-8:
        return 0.0

    mean_excess = np.mean(excess_rets)
    return float((mean_excess / downside_std) * np.sqrt(annualization_factor))


def calculate_max_drawdown_pct(equity_series: pd.Series) -> float:
    """
    Computes maximum peak-to-trough percentage drawdown from an equity curve.
    """
    if equity_series.empty or len(equity_series) < 2:
        return 0.0

    clean_eq = equity_series.dropna().values
    if len(clean_eq) < 2:
        return 0.0

    peak = np.maximum.accumulate(clean_eq)
    drawdowns = (peak - clean_eq) / np.where(peak > 0, peak, 1.0)
    max_dd = np.max(drawdowns) * 100.0
    return float(max_dd) if np.isfinite(max_dd) else 0.0


def calculate_calmar_ratio(
    equity_series: pd.Series,
    initial_capital: float,
    annualization_factor: float = 252.0,
) -> float:
    """
    Computes Calmar Ratio: Annualized ROI / Maximum Drawdown.
    """
    if equity_series.empty or len(equity_series) < 3:
        return 0.0

    max_dd = calculate_max_drawdown_pct(equity_series)
    if max_dd <= 1e-4:
        return 0.0

    total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital
    n_days = max(1, len(equity_series.resample("D").last().dropna()) if isinstance(equity_series.index, pd.DatetimeIndex) else len(equity_series))
    annualized_return = total_return * (annualization_factor / n_days) * 100.0

    return float(annualized_return / max_dd)


def calculate_bar_level_sharpe(
    bar_returns: np.ndarray,
    bars_per_day: float = 25.0,  # e.g., 25 bars per 6.25-hour day for 15m timeframe
    risk_free_rate: float = 0.0,
) -> float:
    """
    Diagnostic Frequency-Aware Bar-Level Sharpe Ratio:
        Bar Sharpe = mean(bar_ret) / std(bar_ret) * sqrt(252 * bars_per_day)
    """
    clean_rets = bar_returns[np.isfinite(bar_returns)]
    if len(clean_rets) < 5:
        return 0.0

    std_dev = np.std(clean_rets, ddof=1)
    if std_dev < 1e-8:
        return 0.0

    ann_multiplier = np.sqrt(252.0 * bars_per_day)
    return float((np.mean(clean_rets) / std_dev) * ann_multiplier)


def calculate_trade_level_metrics(
    net_pnls: List[float],
    initial_capital: float = 500000.0,
) -> Dict[str, Any]:
    """
    Calculates discrete trade-level statistics.
    NOTE: Trade-level metrics are strictly distinct from Daily Annualized Sharpe.
    """
    if not net_pnls:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "net_profit_factor": 0.0,
            "trade_expectancy_inr": 0.0,
            "trade_expectancy_pct": 0.0,
            "win_loss_payoff_ratio": 0.0,
        }

    pnl_arr = np.array(net_pnls)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]

    n_total = len(pnl_arr)
    n_wins = len(wins)
    n_losses = len(losses)

    win_rate = (n_wins / n_total * 100.0) if n_total > 0 else 0.0
    gross_win = float(np.sum(wins)) if n_wins > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if n_losses > 0 else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if n_losses > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    expectancy_inr = float(np.mean(pnl_arr))
    expectancy_pct = (expectancy_inr / initial_capital) * 100.0

    return {
        "total_trades": n_total,
        "winning_trades": n_wins,
        "losing_trades": n_losses,
        "win_rate_pct": round(win_rate, 2),
        "net_profit_factor": round(profit_factor, 2),
        "trade_expectancy_inr": round(expectancy_inr, 2),
        "trade_expectancy_pct": round(expectancy_pct, 4),
        "win_loss_payoff_ratio": round(payoff_ratio, 2),
        "avg_win_inr": round(avg_win, 2),
        "avg_loss_inr": round(avg_loss, 2),
    }
