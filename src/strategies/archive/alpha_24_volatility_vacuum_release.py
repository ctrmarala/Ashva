"""
Ashva Quantitative Alpha 24: Volatility Vacuum Release
Hypothesis:
    When a liquid stock experiences an unusually tight intraday price compression
    (volatility vacuum) followed by a confirmed range expansion bar accompanied by
    elevated relative volume, the first confirmed expansion tends to exhibit short-term
    directional continuation.

Mechanism:
    State transition from low-volatility consolidation to sudden high-volatility momentum.
    Enters in the direction of the confirmed expansion without pre-specifying time-of-day.
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


class Alpha24VolatilityVacuumRelease(BaseHypothesis):
    """
    Alpha 24: Intraday Volatility Vacuum Release Strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        meta = HypothesisMetadata(
            hypothesis_id="alpha_24",
            name="ALPHA_24_VOLATILITY_VACUUM_RELEASE",
            category="VOLATILITY_EXPANSION",
            economic_rationale=(
                "Intraday order flow compression dries up local liquidity. When a sudden "
                "volume-backed price expansion occurs, aggressive market orders breach the "
                "vacuum boundary, creating short-term directional drift before mean reversion."
            ),
            target_instruments=["NIFTY50_LIQUID"],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.VOLATILITY,
        )
        super().__init__(meta, parameters)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "compression_bars": [3, 4, 6],
            "max_compression_atr_ratio": [0.25, 0.35, 0.45],
            "min_expansion_ratio": [1.25, 1.50, 1.75],
            "min_rvol": [1.20, 1.30, 1.50],
            "target_rr": [1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates dense directional signals (+1.0 LONG, -1.0 SHORT, 0.0 FLAT/EXIT).
        """
        out = df.copy()

        # Ensure index is DatetimeIndex
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                out.index = pd.to_datetime(out.index)

        dates = out.index.date
        times = out.index.time

        # 1. Compute Daily ATR (14-day) anchored to prior days to prevent lookahead
        daily_df = out.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(daily_df) >= 14:
            daily_atr_df = TechnicalIndicators.add_atr(daily_df, period=14)
            daily_atr_prev = daily_atr_df["atr_14"].shift(1)
            atr_map = daily_atr_prev.to_dict()
            out["daily_atr"] = [atr_map.get(pd.Timestamp(d), np.nan) for d in dates]
        else:
            out["daily_atr"] = (out["high"] - out["low"]).rolling(14).mean()

        out["daily_atr"] = out["daily_atr"].ffill().bfill()

        # 2. Time-of-Day Mean Volume Baseline (Rolling 20 days shifted by 1 day)
        tod_rolling = out.groupby(times)["volume"].transform(
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

        # Strategy Hyperparameters
        comp_bars = int(self.parameters.get("compression_bars", 4))
        max_comp_atr = float(self.parameters.get("max_compression_atr_ratio", 0.35))
        min_expansion = float(self.parameters.get("min_expansion_ratio", 1.50))
        min_rvol = float(self.parameters.get("min_rvol", 1.30))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        stop_mode = self.parameters.get("stop_mode", "midpoint")

        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(comp_bars, n):
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
                    rationales[i] = "Alpha 24 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across holding bars
            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today:
                continue

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]

            if pd.isna(c_atr) or c_atr <= 0:
                continue

            # Ensure all previous compression bars belong to the SAME trading session
            prior_dates = dates[i - comp_bars : i]
            if not all(d == bar_date for d in prior_dates):
                continue

            # -------------------------------------------------------------
            # 1. Establish Recent Compression Window (Bars i-comp_bars to i-1)
            # -------------------------------------------------------------
            prior_highs = highs[i - comp_bars : i]
            prior_lows = lows[i - comp_bars : i]

            comp_high = float(np.max(prior_highs))
            comp_low = float(np.min(prior_lows))
            comp_range = comp_high - comp_low

            # Verify Volatility Vacuum (Compression Condition)
            if comp_range <= 0.01 or comp_range > (max_comp_atr * c_atr):
                continue

            comp_mid = (comp_high + comp_low) / 2.0
            bar_range = c_high - c_low
            rvol = c_vol / max(1.0, c_tod)

            # Verify Expansion Bar & Volume Confirmation
            is_expansion = (bar_range >= min_expansion * comp_range)
            is_volume_confirmed = (rvol >= min_rvol)

            if not (is_expansion and is_volume_confirmed):
                continue

            # -------------------------------------------------------------
            # 2. Evaluate Directional Vacuum Release
            # -------------------------------------------------------------
            # Bullish Vacuum Release (LONG)
            if (c_close > comp_high) and (c_close > c_open):
                curr_state = 1.0
                sl_ref = comp_mid if stop_mode == "midpoint" else comp_low
                stop_dist = max(c_close - sl_ref, 0.20 * c_atr)
                curr_sl = c_close - stop_dist
                curr_tp = c_close + (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 24 VACUUM LONG: CompRange={comp_range:.1f} ({comp_range/c_atr:.2f} ATR) | "
                    f"ExpansionBar={bar_range:.1f} | Close={c_close:.1f} > CompHigh={comp_high:.1f} | "
                    f"RVOL={rvol:.2f}x | SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                )
                signals[i] = 1.0
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                traded_today = True

            # Bearish Vacuum Release (SHORT)
            elif (c_close < comp_low) and (c_close < c_open):
                curr_state = -1.0
                sl_ref = comp_mid if stop_mode == "midpoint" else comp_high
                stop_dist = max(sl_ref - c_close, 0.20 * c_atr)
                curr_sl = c_close + stop_dist
                curr_tp = c_close - (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 24 VACUUM SHORT: CompRange={comp_range:.1f} ({comp_range/c_atr:.2f} ATR) | "
                    f"ExpansionBar={bar_range:.1f} | Close={c_close:.1f} < CompLow={comp_low:.1f} | "
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
