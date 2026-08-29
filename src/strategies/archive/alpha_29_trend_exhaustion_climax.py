"""
Ashva Quantitative Alpha 29: Multi-Timeframe Trend Exhaustion Climax Reversal
Hypothesis:
    When a liquid stock experiences a vertical exhaustion impulse that extends price
    substantially away from its intraday VWAP (> 1.8x Daily ATR) accompanied by an
    extreme volume climax (> 2.0x time-of-day baseline), institutional market makers
    absorb the liquidity panic, triggering a high-probability mean-reverting pullback
    back towards the intraday VWAP.

Mechanism:
    Order book exhaustion climax and mean reversion to intraday VWAP.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.research.hypothesis import (
    BaseHypothesis,
    HypothesisMetadata,
    StrategyHorizon,
    MarketMechanism,
)
from src.features.indicators import TechnicalIndicators


class Alpha29TrendExhaustionClimax(BaseHypothesis):
    """
    Alpha 29: Trend Exhaustion Climax Mean Reversion Strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        meta = HypothesisMetadata(
            hypothesis_id="alpha_29",
            name="ALPHA_29_TREND_EXHAUSTION_CLIMAX",
            category="MEAN_REVERSION_CLIMAX",
            economic_rationale=(
                "Vertical price spikes accompanied by massive volume spikes represent retail "
                "liquidity chasing or stop runs that exhaust aggressive order flow. "
                "Passive institutional liquidity absorbs the move, causing rapid reversion to fair-value VWAP."
            ),
            target_instruments=["NIFTY50_LIQUID"],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
        )
        super().__init__(meta, parameters)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_vwap_dist_atr": [0.55, 0.65, 0.75],
            "min_climax_rvol": [1.30, 1.50, 1.80],
            "target_rr": [1.25, 1.50, 2.00],
            "max_entry_hour": [13, 14],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                out.index = pd.to_datetime(out.index)

        dates = out.index.date
        times = out.index.time

        # 1. Daily ATR (14-day) anchored to prior days
        daily_df = out.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(daily_df) >= 14:
            daily_atr_df = TechnicalIndicators.add_atr(daily_df, period=14)
            daily_atr_prev = daily_atr_df["atr_14"].shift(1)
            atr_map = daily_atr_prev.to_dict()
            out["daily_atr"] = [atr_map.get(pd.Timestamp(d), np.nan) for d in dates]
        else:
            out["daily_atr"] = (out["high"] - out["low"]).rolling(14).mean()

        out["daily_atr"] = out["daily_atr"].ffill().bfill()

        # 2. Intraday Anchored VWAP
        typical_p = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_p * out["volume"]
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_v"] = out["volume"].groupby(dates).cumsum()
        out["vwap"] = (out["cum_pv"] / out["cum_v"].replace(0, np.nan)).bfill().ffill()

        # 3. Time-of-Day Mean Volume Baseline
        tod_rolling = out.groupby(times)["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_rolling

        # Strategy Hyperparameters
        min_dist_atr = float(self.parameters.get("min_vwap_dist_atr", 0.65))
        min_climax_rvol = float(self.parameters.get("min_climax_rvol", 1.40))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_entry_hour = int(self.parameters.get("max_entry_hour", 14))

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
        vwaps = out["vwap"].values

        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_0930 = pd.to_datetime("09:30:00").time()
        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(1, n):
            bar_date = dates[i]
            bar_time = times[i]

            # Reset on new trading day
            if bar_date != current_day:
                current_day = bar_date
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
                    rationales[i] = "Alpha 29 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across holding bars
            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or bar_time < t_0930 or bar_time.hour > max_entry_hour:
                continue

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]
            c_vwap = vwaps[i]

            if pd.isna(c_atr) or c_atr <= 0 or pd.isna(c_vwap):
                continue

            rvol = c_vol / max(1.0, c_tod)
            if rvol < min_climax_rvol:
                continue

            vwap_dist = abs(c_close - c_vwap)
            required_dist = min_dist_atr * c_atr

            if vwap_dist < required_dist:
                continue

            # -------------------------------------------------------------
            # Case 1: Bullish Climax Over-Extension -> Enter SHORT (Fade Top)
            # -------------------------------------------------------------
            # Upper wick rejection or overbought exhaustion above VWAP
            if (c_close > c_vwap):
                upper_wick = c_high - max(c_close, c_open)
                bar_range = c_high - c_low
                # Verify upper rejection wick or extreme extension
                if upper_wick >= 0.35 * bar_range or (c_close < c_open):
                    curr_state = -1.0
                    stop_dist = max(c_high - c_close + 0.15 * c_atr, 0.30 * c_atr)
                    curr_sl = c_close + stop_dist
                    curr_tp = max(c_vwap, c_close - (target_rr * stop_dist))
                    curr_rationale = (
                        f"Alpha 29 SHORT CLIMAX: Close={c_close:.1f} vs VWAP={c_vwap:.1f} (+{vwap_dist/c_atr:.2f} ATR) | "
                        f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f}"
                    )
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

            # -------------------------------------------------------------
            # Case 2: Bearish Climax Over-Extension -> Enter LONG (Fade Bottom)
            # -------------------------------------------------------------
            elif (c_close < c_vwap):
                lower_wick = min(c_close, c_open) - c_low
                bar_range = c_high - c_low
                if lower_wick >= 0.35 * bar_range or (c_close > c_open):
                    curr_state = 1.0
                    stop_dist = max(c_close - c_low + 0.15 * c_atr, 0.30 * c_atr)
                    curr_sl = c_close - stop_dist
                    curr_tp = min(c_vwap, c_close + (target_rr * stop_dist))
                    curr_rationale = (
                        f"Alpha 29 LONG CLIMAX: Close={c_close:.1f} vs VWAP={c_vwap:.1f} (-{vwap_dist/c_atr:.2f} ATR) | "
                        f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f}"
                    )
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = curr_rationale
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
