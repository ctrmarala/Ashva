"""
Ashva Quantitative Strategy: Statistical Range Reversion - Long Boundary (Alpha 10)
Captures multi-day swing mean-reversion in non-trending equities oscillating within
statistically stable, defended 20-day boundary channels.

Hypothesis:
When a liquid equity establishes a stable 20-day trading range during a non-trending regime (ADX_14 < 22),
a defended test of the lower range boundary with a bullish rejection candle offers an asymmetric mean-reversion
opportunity toward the 20-day range midpoint, provided the expected reward-to-risk ratio to the midpoint is >= 1.25R.
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


class Alpha10StatisticalRangeReversion(BaseHypothesis):
    """
    Statistical Range Reversion - Long Boundary (Alpha 10 - Multi-Day Swing):
    1. 20-day Range: Upper = max(High_20d), Lower = min(Low_20d), Midpoint = (Upper + Lower) / 2.
    2. Range Stability Gate: 4% <= Range_Width / Close <= 18%, Daily ADX(14) < 22.0.
    3. Defended Lower Boundary: Low tests Lower_20d + 0.35 * ATR, holds >= Lower_20d - 0.25 * ATR.
    4. Bullish Rejection: Close > Open with Lower Wick >= 25% of candle range.
    5. Reward-to-Risk Gate: (Midpoint - Close) / (Close - Stop) >= 1.25R.
    6. Multi-Day Execution: Next-bar open fill, Stop = Lower_20d - 0.50 * ATR, Target = Midpoint, Max Hold = 8 Days.
    7. Full Delivery Costs: 0.10% Buy STT + 0.10% Sell STT + Delivery friction.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_10_STATISTICAL_RANGE_REVERSION",
            name="Alpha_10_Statistical_Range_Reversion",
            category="RANGE_MEAN_REVERSION_SWING",
            economic_rationale=(
                "When a liquid equity establishes a stable 20-day trading range during a non-trending regime (ADX < 22), "
                "a defended test of the lower range boundary with a bullish rejection candle offers an asymmetric mean-reversion "
                "opportunity toward the 20-day range midpoint, provided the expected reward-to-risk ratio to the midpoint is >= 1.25R."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.SWING,
            mechanism=MarketMechanism.RANGE,
            author="AshvaQuantLab",
        )
        params = parameters or {
            "range_window_days": 20,           # 20-day historical boundary window
            "max_daily_adx": 22.0,             # Non-trending regime gate (Daily ADX < 22)
            "min_range_width_pct": 0.04,       # Min range width (4% of price)
            "max_range_width_pct": 0.18,       # Max range width (18% of price)
            "boundary_touch_atr_mult": 0.35,   # Proximity to lower boundary <= 0.35 * Daily ATR
            "stop_buffer_atr_mult": 0.50,      # Stop placed at Lower Boundary - 0.50 * Daily ATR
            "min_wick_ratio": 0.25,            # Rejection wick >= 25% of candle range
            "min_reward_risk_ratio": 1.25,     # Reward to Midpoint / Risk >= 1.25
            "max_hold_bars": 200,              # Max holding duration: ~8 trading days (200 15m bars)
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "range_window_days": [15, 20, 25],
            "max_daily_adx": [20.0, 22.0, 25.0],
            "min_reward_risk_ratio": [1.20, 1.25, 1.35],
            "stop_buffer_atr_mult": [0.40, 0.50, 0.60],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 10 (Multi-Day Swing Long Boundary).
        """
        out = df.copy()

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date

        # 1. Build Daily Canvas strictly from completed prior sessions (Shifted 1 session)
        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last")
        )

        range_win = int(self.parameters.get("range_window_days", 20))
        max_adx = float(self.parameters.get("max_daily_adx", 22.0))
        min_width = float(self.parameters.get("min_range_width_pct", 0.04))
        max_width = float(self.parameters.get("max_range_width_pct", 0.18))
        touch_mult = float(self.parameters.get("boundary_touch_atr_mult", 0.35))
        stop_mult = float(self.parameters.get("stop_buffer_atr_mult", 0.50))
        min_wick = float(self.parameters.get("min_wick_ratio", 0.25))
        min_rr = float(self.parameters.get("min_reward_risk_ratio", 1.25))
        max_hold = int(self.parameters.get("max_hold_bars", 200))

        # Rolling 20-Day Range Boundaries (Shifted 1 session to prevent look-ahead bias)
        daily_high_20d = daily_summary["day_high"].rolling(range_win, min_periods=range_win).max().shift(1)
        daily_low_20d = daily_summary["day_low"].rolling(range_win, min_periods=range_win).min().shift(1)
        daily_mid_20d = (daily_high_20d + daily_low_20d) / 2.0

        # Daily ATR(14) (Shifted 1 session)
        prev_close = daily_summary["day_close"].shift(1)
        tr1 = daily_summary["day_high"] - daily_summary["day_low"]
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)

        # Daily ADX(14) (Shifted 1 session)
        daily_adx_df = TI.add_adx(
            daily_summary.rename(columns={"day_high": "high", "day_low": "low", "day_close": "close"}),
            period=14
        )
        daily_adx14 = daily_adx_df["adx_14"].shift(1)

        # Map back to 15m intraday bars
        out["range_high_20d"] = pd.Series(dates, index=out.index).map(daily_high_20d).ffill()
        out["range_low_20d"] = pd.Series(dates, index=out.index).map(daily_low_20d).ffill()
        out["range_mid_20d"] = pd.Series(dates, index=out.index).map(daily_mid_20d).ffill()
        out["daily_atr"] = pd.Series(dates, index=out.index).map(daily_atr14).ffill()
        out["daily_adx"] = pd.Series(dates, index=out.index).map(daily_adx14).ffill()

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        range_highs = out["range_high_20d"].values
        range_lows = out["range_low_20d"].values
        range_mids = out["range_mid_20d"].values
        daily_atrs = out["daily_atr"].values
        daily_adxs = out["daily_adx"].values

        in_swing_trade = False
        bars_in_trade = 0
        active_sl = 0.0
        active_tp = 0.0

        for i in range(1, n):
            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_rhigh = range_highs[i]
            c_rlow = range_lows[i]
            c_rmid = range_mids[i]
            c_atr = daily_atrs[i]
            c_adx = daily_adxs[i]

            # -------------------------------------------------------------
            # Manage Active Multi-Day Swing Trade
            # -------------------------------------------------------------
            if in_swing_trade:
                bars_in_trade += 1

                # Check SL / TP / Max Hold Duration
                hit_sl = (c_low <= active_sl)
                hit_tp = (c_high >= active_tp)
                time_expired = (bars_in_trade >= max_hold)

                if hit_sl or hit_tp or time_expired:
                    in_swing_trade = False
                    bars_in_trade = 0
                    active_sl = 0.0
                    active_tp = 0.0
                    signals[i] = 0.0
                    rationales[i] = (
                        "Alpha 10 SWING EXIT: "
                        + ("Take Profit Hit" if hit_tp else ("Stop Loss Hit" if hit_sl else "Max 8-Day Duration Reached"))
                    )
                else:
                    signals[i] = 0.0  # Holding swing position across days
                    stop_loss[i] = active_sl
                    take_profit[i] = active_tp
                continue

            # -------------------------------------------------------------
            # Evaluate New Swing Entry (LONG Boundary Reversion)
            # -------------------------------------------------------------
            if pd.isna(c_rlow) or pd.isna(c_rhigh) or pd.isna(c_rmid) or pd.isna(c_atr) or pd.isna(c_adx) or c_close <= 0:
                continue

            # 1. Range Stability & Non-Trending Gates
            range_width = c_rhigh - c_rlow
            width_ratio = range_width / c_close

            if not (min_width <= width_ratio <= max_width):
                continue

            if c_adx >= max_adx:
                continue

            # 2. Defended Lower Boundary Test
            proximity_zone = c_rlow + (touch_mult * c_atr)
            lower_floor = c_rlow - (0.25 * c_atr)

            if (c_low <= proximity_zone) and (c_low >= lower_floor):
                candle_range = max(c_high - c_low, 0.01)
                lower_wick = min(c_open, c_close) - c_low
                lower_wick_ratio = lower_wick / candle_range

                # 3. Bullish Rejection Confirmation
                if (c_close > c_open) and (lower_wick_ratio >= min_wick) and (c_close < c_rmid):
                    sl_price = c_rlow - (stop_mult * c_atr)
                    risk_dist = c_close - sl_price
                    reward_dist = c_rmid - c_close

                    if risk_dist > 0.05:
                        expected_rr = reward_dist / risk_dist

                        # 4. Reward-to-Risk Gate (>= 1.25R to Midpoint)
                        if expected_rr >= min_rr:
                            # Discrete Swing Entry Pulse (Next-Bar Open Fill)
                            signals[i] = 1.0
                            stop_loss[i] = sl_price
                            take_profit[i] = c_rmid
                            active_sl = sl_price
                            active_tp = c_rmid
                            in_swing_trade = True
                            bars_in_trade = 0
                            rationales[i] = (
                                f"Alpha 10 SWING LONG: 20D_Low=Rs {c_rlow:.1f} | ADX={c_adx:.1f} | "
                                f"LWick={lower_wick_ratio*100:.1f}% | SL=Rs {sl_price:.1f} | TP=Rs {c_rmid:.1f} (RR={expected_rr:.2f}R)"
                            )

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
