"""
Ashva Quantitative Strategy: Overnight Gap Momentum Drift (Alpha 14 - Intraday Gap Expansion)
Captures persistent institutional order-flow continuation following overnight price/information imbalances.

Hypothesis:
When a liquid equity opens with a moderate overnight gap (>= 0.30%) and the opening 15-minute bar confirms
directional acceptance (Body >= 50% in gap direction with elevated volume RVOL >= 1.25x), the overnight
institutional order flow drives persistent intraday continuation toward a 1.5R target.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators as TI
from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    HypothesisStatus,
    StrategyHorizon,
    MarketMechanism,
)


class Alpha14GapMomentumDrift(BaseHypothesis):
    """
    Overnight Gap Momentum Drift (Alpha 14):
    1. Overnight Gap: |Open_09:15 - Prev_Close| / Prev_Close >= min_gap_pct (0.30%).
    2. Confirmation Bar (09:15-09:30 IST):
       - Gap Up: Close > Open, Body / Range >= 50%, RVOL >= 1.25x.
       - Gap Down: Close < Open, Body / Range >= 50%, RVOL >= 1.25x.
    3. Execution & Sizing: Entry at 09:30 Open, Stop at Bar 1 Extreme (Low for Long, High for Short), Target = 1.50R, 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_14_GAP_MOMENTUM_DRIFT",
            name="Alpha_14_Gap_Momentum_Drift",
            category="OVERNIGHT_GAP_MOMENTUM",
            economic_rationale=(
                "When a liquid equity opens with a moderate overnight gap (>= 0.30%) and the opening "
                "15-minute bar confirms directional acceptance (Body >= 50% with RVOL >= 1.25x), "
                "institutional portfolio rebalancing and overnight news drift drive persistent intraday continuation."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MOMENTUM,
            author="AshvaQuantLab",
        )
        params = parameters or {
            "min_gap_pct": 0.0030,            # 0.30% minimum overnight gap
            "min_body_ratio": 0.50,            # 50% minimum candle body ratio
            "min_rvol": 1.25,                  # 1.25x minimum RVOL
            "target_rr": 1.50,                 # 1.50R target multiple
            "max_gap_atr_ratio": 0.80,         # Gap <= 0.80 * Daily ATR (prevent exhaustion gaps)
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_gap_pct": [0.0025, 0.0030, 0.0040],
            "min_body_ratio": [0.45, 0.50, 0.60],
            "min_rvol": [1.15, 1.25, 1.35],
            "target_rr": [1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 14.
        """
        out = df.copy()

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        # 1. Build Daily Canvas strictly from completed prior sessions (Shifted 1 session)
        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last")
        )

        prev_close = daily_summary["day_close"].shift(1)
        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)
        prev_prev_close = daily_summary["day_close"].shift(2)

        tr1 = prev_high - prev_low
        tr2 = (prev_high - prev_prev_close).abs()
        tr3 = (prev_low - prev_prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean()

        out["prev_day_close"] = pd.Series(dates, index=out.index).map(prev_close).ffill()
        out["daily_atr"] = pd.Series(dates, index=out.index).map(daily_atr14).ffill()

        # 2. 20-Session TOD Rolling Volume Baseline (Shifted 1 session)
        tod_rolling = out.groupby("time_str")["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_rolling

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        volumes = out["volume"].values
        tod_vols = out["tod_mean_vol"].values
        prev_closes = out["prev_day_close"].values
        daily_atrs = out["daily_atr"].values

        min_gap = float(self.parameters.get("min_gap_pct", 0.0030))
        min_body = float(self.parameters.get("min_body_ratio", 0.50))
        min_rvol = float(self.parameters.get("min_rvol", 1.25))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_gap_atr = float(self.parameters.get("max_gap_atr_ratio", 0.80))

        current_day = None
        traded_today = False

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            if bar_date != current_day:
                current_day = bar_date
                traded_today = False

            if traded_today:
                continue

            # Evaluate strictly on Bar 1 (09:15 to 09:30)
            if hour == 9 and minute == 15:
                p_close = prev_closes[i]
                c_atr = daily_atrs[i]
                c_open = opens[i]
                c_close = closes[i]
                c_high = highs[i]
                c_low = lows[i]
                c_vol = volumes[i]
                c_tod = tod_vols[i]

                if pd.isna(p_close) or p_close <= 0 or pd.isna(c_atr) or c_atr <= 0:
                    continue

                gap_pct = (c_open - p_close) / p_close
                gap_abs = abs(c_open - p_close)

                # Filter out exhaustion gaps
                if gap_abs > (max_gap_atr * c_atr):
                    continue

                candle_range = max(c_high - c_low, 0.01)
                body_ratio = abs(c_close - c_open) / candle_range
                rvol = c_vol / max(1.0, c_tod)

                # Bullish Gap & Continuation (LONG)
                if (gap_pct >= min_gap) and (c_close > c_open) and (body_ratio >= min_body) and (rvol >= min_rvol):
                    signals[i] = 1.0
                    sl_price = c_low
                    stop_dist = max(c_close - sl_price, 0.15 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 14 GAP LONG: Gap=+{gap_pct*100:.2f}% | Body={body_ratio*100:.1f}% | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

                # Bearish Gap & Continuation (SHORT)
                elif (gap_pct <= -min_gap) and (c_close < c_open) and (body_ratio >= min_body) and (rvol >= min_rvol):
                    signals[i] = -1.0
                    sl_price = c_high
                    stop_dist = max(sl_price - c_close, 0.15 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 14 GAP SHORT: Gap={gap_pct*100:.2f}% | Body={body_ratio*100:.1f}% | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
