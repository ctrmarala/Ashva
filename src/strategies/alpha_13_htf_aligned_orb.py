"""
Ashva Quantitative Strategy: Higher-Timeframe Trend-Aligned Opening Momentum (Alpha 13 - HTF-ORB)
Captures high-conviction opening range breakouts that strictly align with the multi-day macro trend.

Hypothesis:
An opening range breakout (ORB) succeeds with high expectancy when it aligns with the higher-timeframe
institutional trend (Daily Close > Daily EMA20 and Prior Day Bullish Direction). By filtering out counter-trend
breakouts, the strategy eliminates chop and captures large directional continuation legs to a 2.0R target.
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


class Alpha13HTFAlignedORB(BaseHypothesis):
    """
    Higher-Timeframe Trend-Aligned Opening Momentum (Alpha 13):
    1. Higher-Timeframe (Daily) Filter (Shifted 1 session):
       - Bullish Regime: Prior Day Close > Prior Day Open AND Prior Day Close > 20-day Daily EMA.
       - Bearish Regime: Prior Day Close < Prior Day Open AND Prior Day Close < 20-day Daily EMA.
    2. Opening Range (09:15-09:30 IST): High, Low, Range.
    3. Breakout Window (09:30-10:30 IST):
       - Long: 15m Close > OR15 High + Bullish Candle + RVOL >= 1.20x (in Bullish HTF Regime).
       - Short: 15m Close < OR15 Low + Bearish Candle + RVOL >= 1.20x (in Bearish HTF Regime).
    4. Risk & Execution: Fill at next-bar open, Stop at OR15 Midpoint (tighter risk = higher R multiple), Target = 2.0R, 15:15 EOD exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_13_HTF_ALIGNED_ORB",
            name="Alpha_13_HTF_Aligned_ORB",
            category="MULTI_TIMEFRAME_MOMENTUM",
            economic_rationale=(
                "An opening range breakout succeeds with high expectancy when it aligns with the "
                "higher-timeframe institutional trend (Daily Close > Daily EMA20 and Prior Day Bullish Direction). "
                "By filtering out counter-trend breakouts, the strategy eliminates low-conviction chop and "
                "captures large directional trend expansion to a 2.0R target."
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
            "min_rvol": 1.20,                  # Breakout volume >= 1.20x shifted TOD baseline
            "target_rr": 2.00,                 # 2.0R target multiple
            "max_or_atr_ratio": 0.50,          # OR15 range <= 0.50 * Daily ATR
            "stop_mode": "midpoint",           # "midpoint" or "opposite"
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.10, 1.20, 1.30],
            "target_rr": [1.50, 2.00, 2.50],
            "max_or_atr_ratio": [0.40, 0.50, 0.60],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 13.
        """
        out = df.copy()

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        # 1. Build Daily Canvas strictly from completed prior sessions (Shifted 1 session)
        daily_summary = out.groupby(dates).agg(
            day_open=("open", "first"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last")
        )

        prev_open = daily_summary["day_open"].shift(1)
        prev_close = daily_summary["day_close"].shift(1)
        prev_high = daily_summary["day_high"].shift(1)
        prev_low = daily_summary["day_low"].shift(1)

        # Daily EMA20 (Shifted 1 session)
        daily_ema20 = daily_summary["day_close"].ewm(span=20, adjust=False).mean().shift(1)

        # Daily ATR(14) (Shifted 1 session)
        prev_prev_close = daily_summary["day_close"].shift(2)
        tr1 = prev_high - prev_low
        tr2 = (prev_high - prev_prev_close).abs()
        tr3 = (prev_low - prev_prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean()

        # HTF Trend Regime: +1 = Bullish Trend, -1 = Bearish Trend, 0 = Mixed
        htf_trend = pd.Series(0, index=daily_summary.index)
        bullish_mask = (prev_close > prev_open) & (prev_close > daily_ema20)
        bearish_mask = (prev_close < prev_open) & (prev_close < daily_ema20)
        htf_trend[bullish_mask] = 1
        htf_trend[bearish_mask] = -1

        out["daily_atr"] = pd.Series(dates, index=out.index).map(daily_atr14).ffill()
        out["htf_trend"] = pd.Series(dates, index=out.index).map(htf_trend).ffill()

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
        htf_trends = out["htf_trend"].values

        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 2.00))
        max_or_atr = float(self.parameters.get("max_or_atr_ratio", 0.50))
        stop_mode = self.parameters.get("stop_mode", "midpoint")

        current_day = None
        or_high = 0.0
        or_low = 0.0
        or_established = False
        traded_today = False

        for i in range(n):
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            # Reset on new day
            if bar_date != current_day:
                current_day = bar_date
                or_high = 0.0
                or_low = 0.0
                or_established = False
                traded_today = False

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]
            c_htf = htf_trends[i]

            # -------------------------------------------------------------
            # 1. Establish 09:15 Opening Range (Bar 1)
            # -------------------------------------------------------------
            if hour == 9 and minute == 15:
                or_high = c_high
                or_low = c_low
                or_established = True
                continue

            # -------------------------------------------------------------
            # 2. Breakout Evaluation Window (09:30 to 10:30 IST)
            # -------------------------------------------------------------
            if (not or_established) or traded_today or pd.isna(c_atr) or c_atr <= 0:
                continue

            if (hour == 9 and minute >= 30) or (hour == 10 and minute <= 30):
                or_range = or_high - or_low
                if or_range <= 0.01 or or_range > (max_or_atr * c_atr):
                    continue

                or_mid = (or_high + or_low) / 2.0
                rvol = c_vol / max(1.0, c_tod)

                # Bullish HTF-Aligned Breakout (LONG)
                if (c_htf == 1) and (c_close > or_high) and (c_close > c_open) and (rvol >= min_rvol):
                    signals[i] = 1.0
                    sl_price = or_mid if stop_mode == "midpoint" else or_low
                    stop_dist = max(c_close - sl_price, 0.20 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 13 LONG: HTF Bullish (PDay Close > EMA20) | OR15 Breakout Close={c_close:.1f} > OR_High={or_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

                # Bearish HTF-Aligned Breakdown (SHORT)
                elif (c_htf == -1) and (c_close < or_low) and (c_close < c_open) and (rvol >= min_rvol):
                    signals[i] = -1.0
                    sl_price = or_mid if stop_mode == "midpoint" else or_high
                    stop_dist = max(sl_price - c_close, 0.20 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 13 SHORT: HTF Bearish (PDay Close < EMA20) | OR15 Breakdown Close={c_close:.1f} < OR_Low={or_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
