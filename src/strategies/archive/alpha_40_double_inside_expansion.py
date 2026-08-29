"""
Ashva Quantitative Strategy: Double Inside Day Range Expansion Breakout (Alpha 40)
Captures explosive directional momentum release following a 2-day consecutive inside compression (NR2 in NR3).

Hypothesis:
When an equity forms a Double Inside Day (Day T-1 inside Day T-2, which is also inside Day T-3),
market volatility compression reaches absolute structural tightness. On Day 4, a 15m opening range breakout
with volume confirmation (RVOL >= 1.15x) triggers massive volatility expansion toward a 1.75R target.
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


class Alpha40DoubleInsideExpansion(BaseHypothesis):
    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_40_DOUBLE_INSIDE_EXPANSION",
            name="Alpha_40_Double_Inside_Expansion",
            category="VOLATILITY_CONTRACTION_EXPANSION",
            economic_rationale=(
                "Two consecutive inside sessions create extreme multi-day volatility compression. "
                "An opening range breakout on Day 4 triggers explosive stop-cascades and institutional expansion."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.BREAKOUT,
        )
        default_params = {
            "min_rvol": 1.15,
            "target_rr": 1.75,
            "max_or_atr_ratio": 0.80,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(metadata=meta, parameters=default_params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.10, 1.15, 1.25],
            "target_rr": [1.50, 1.75, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last")
        )

        h = daily_summary["day_high"]
        l = daily_summary["day_low"]

        # Double Inside Day: Day T-1 is inside Day T-2 AND Day T-2 is inside Day T-3
        is_double_inside = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2)) & (h.shift(2) < h.shift(3)) & (l.shift(2) > l.shift(3))

        prev_close = daily_summary["day_close"].shift(1)
        tr1 = daily_summary["day_high"] - daily_summary["day_low"]
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)

        out["is_d_inside"] = pd.Series(dates, index=out.index).map(is_double_inside).ffill().fillna(False)
        out["daily_atr"] = pd.Series(dates, index=out.index).map(daily_atr14).ffill()

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
        daily_atrs = out["daily_atr"].values
        d_inside_flags = out["is_d_inside"].values

        min_rvol = float(self.parameters.get("min_rvol", 1.15))
        target_rr = float(self.parameters.get("target_rr", 1.75))
        max_or_atr = float(self.parameters.get("max_or_atr_ratio", 0.80))

        current_day = None
        or_high = 0.0
        or_low = 0.0
        or_established = False
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_0915 = pd.to_datetime("09:15:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            if bar_date != current_day:
                current_day = bar_date
                or_high = 0.0
                or_low = 0.0
                or_established = False
                traded_today = False
                curr_state = 0.0
                curr_sl = 0.0
                curr_tp = 0.0
                curr_rationale = ""

            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 40 EXIT: 15:15 EOD Square-Off"
                continue

            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]
            is_d_inside_today = d_inside_flags[i]

            if bar_time == t_0915:
                or_high = c_high
                or_low = c_low
                or_established = True
                continue

            if (not or_established) or (not is_d_inside_today) or traded_today or pd.isna(c_atr) or c_atr <= 0:
                continue

            if (hour == 9 and minute >= 30) or (hour == 10 and minute <= 30):
                or_range = or_high - or_low
                if or_range <= 0.01 or or_range > (max_or_atr * c_atr):
                    continue

                rvol = c_vol / max(1.0, c_tod)

                # Long Breakout
                if (c_close > or_high) and (c_close > c_open) and (rvol >= min_rvol):
                    curr_state = 1.0
                    sl_price = or_low
                    stop_dist = max(c_close - sl_price, 0.15 * c_atr)
                    curr_sl = c_close - stop_dist
                    curr_tp = c_close + (target_rr * stop_dist)
                    curr_rationale = f"Alpha 40 LONG: Double Inside Day Breakout Close={c_close:.1f} > OR_H={or_high:.1f} | RVOL={rvol:.2f}x"
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

                # Short Breakout
                elif (c_close < or_low) and (c_close < c_open) and (rvol >= min_rvol):
                    curr_state = -1.0
                    sl_price = or_high
                    stop_dist = max(sl_price - c_close, 0.15 * c_atr)
                    curr_sl = c_close + stop_dist
                    curr_tp = c_close - (target_rr * stop_dist)
                    curr_rationale = f"Alpha 40 SHORT: Double Inside Day Breakout Close={c_close:.1f} < OR_L={or_low:.1f} | RVOL={rvol:.2f}x"
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["entry_rationale"] = rationales
        return out
