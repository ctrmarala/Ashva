"""
Ashva Quantitative Strategy: 3-Day Trend-Aligned Opening Momentum (Alpha 18)
Captures persistent institutional order-flow continuation in stocks exhibiting 3 consecutive sessions of higher lows or lower highs.

Hypothesis:
When a liquid equity has posted 3 consecutive sessions of higher daily lows (Low(T-1) > Low(T-2) > Low(T-3)),
directional institutional flow is established. An opening range breakout on Day T aligned with this 3-day trend
has a high probability of large trend expansion to a 1.50R target.
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


class Alpha18ThreeDayTrendORB(BaseHypothesis):
    """
    3-Day Trend-Aligned Opening Momentum (Alpha 18):
    1. 3-Day Momentum Filter (Shifted 1 session):
       - Bullish 3-Day Trend: Low(T-1) > Low(T-2) > Low(T-3).
       - Bearish 3-Day Trend: High(T-1) < High(T-2) < High(T-3).
    2. Opening Range (09:15-09:30 IST): High, Low, Range.
    3. Breakout Window (09:30-10:30 IST):
       - Long: 15m Close > OR15 High + Bullish Candle + RVOL >= 1.20x (in 3-Day Bullish Trend).
       - Short: 15m Close < OR15 Low + Bearish Candle + RVOL >= 1.20x (in 3-Day Bearish Trend).
    4. Execution & Risk: Next-bar open fill, Stop at OR15 Midpoint, Target = 1.50R, 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_18_THREE_DAY_TREND_ORB",
            name="Alpha_18_Three_Day_Trend_ORB",
            category="MULTI_SESSION_MOMENTUM",
            economic_rationale=(
                "3 consecutive sessions of directional price structure establish persistent institutional order flow. "
                "Opening range breakouts aligned with this 3-day momentum leg exhibit clean directional expansion."
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
            "min_rvol": 1.20,                  # Volume >= 1.20x shifted TOD baseline
            "target_rr": 1.50,                 # 1.50R target multiple
            "max_or_atr_ratio": 0.50,          # OR15 range <= 0.50 * Daily ATR
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.10, 1.20, 1.30],
            "target_rr": [1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 18.
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

        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)
        prev2_high = daily_summary["day_high"].shift(2)
        prev2_low = daily_summary["day_low"].shift(2)
        prev3_high = daily_summary["day_high"].shift(3)
        prev3_low = daily_summary["day_low"].shift(3)

        # 3-Day Momentum
        is_3d_bullish = (prev_low > prev2_low) & (prev2_low > prev3_low)
        is_3d_bearish = (prev_high < prev2_high) & (prev2_high < prev3_high)

        trend_3d = pd.Series(0, index=daily_summary.index)
        trend_3d[is_3d_bullish] = 1
        trend_3d[is_3d_bearish] = -1

        # Daily ATR(14) (Shifted 1 session)
        prev_close = daily_summary["day_close"].shift(1)
        prev_prev_close = daily_summary["day_close"].shift(2)
        tr1 = prev_high - prev_low
        tr2 = (prev_high - prev_prev_close).abs()
        tr3 = (prev_low - prev_prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean()

        out["trend_3d"] = pd.Series(dates, index=out.index).map(trend_3d).ffill()
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
        daily_atrs = out["daily_atr"].values
        trends_3d = out["trend_3d"].values

        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_or_atr = float(self.parameters.get("max_or_atr_ratio", 0.50))

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

            # Intraday 15:15 EOD Square-Off
            if bar_time >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Alpha 18 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across intraday bars
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
            c_trend = trends_3d[i]

            # -------------------------------------------------------------
            # 1. Establish 09:15 Opening Range (Bar 1)
            # -------------------------------------------------------------
            if bar_time == t_0915:
                or_high = c_high
                or_low = c_low
                or_established = True
                continue

            # -------------------------------------------------------------
            # 2. Breakout Evaluation Window (09:30 to 10:30 IST)
            # -------------------------------------------------------------
            if (not or_established) or traded_today or pd.isna(c_atr) or c_atr <= 0 or c_trend == 0:
                continue

            if (hour == 9 and minute >= 30) or (hour == 10 and minute <= 30):
                or_range = or_high - or_low
                if or_range <= 0.01 or or_range > (max_or_atr * c_atr):
                    continue

                or_mid = (or_high + or_low) / 2.0
                rvol = c_vol / max(1.0, c_tod)

                # Bullish 3-Day Trend Aligned Breakout (LONG)
                if (c_trend == 1) and (c_close > or_high) and (c_close > c_open) and (rvol >= min_rvol):
                    curr_state = 1.0
                    stop_dist = max(c_close - or_mid, 0.15 * c_atr)
                    curr_sl = c_close - stop_dist
                    curr_tp = c_close + (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 18 3D LONG: 3 Consecutive Higher Lows | OR15 Breakout Close={c_close:.1f} > OR_High={or_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                    )
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

                # Bearish 3-Day Trend Aligned Breakdown (SHORT)
                elif (c_trend == -1) and (c_close < or_low) and (c_close < c_open) and (rvol >= min_rvol):
                    curr_state = -1.0
                    stop_dist = max(or_mid - c_close, 0.15 * c_atr)
                    curr_sl = c_close + stop_dist
                    curr_tp = c_close - (target_rr * stop_dist)
                    curr_rationale = (
                        f"Alpha 18 3D SHORT: 3 Consecutive Lower Highs | OR15 Breakdown Close={c_close:.1f} < OR_Low={or_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                    )
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
