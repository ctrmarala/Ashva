"""
Ashva Quantitative Strategy: Intraday VWAP Momentum Breakout (Alpha 20 - VWAP-Momentum)
Captures institutional trend continuation when price crosses VWAP with strong volume acceleration.

Hypothesis:
When a liquid equity breaks decisively across VWAP in the morning session (10:00-11:30 IST) with high volume
expansion (RVOL >= 1.30x) in the direction of the daily macro trend, institutional VWAP execution algorithms
aggressively chase liquidity, generating persistent continuation to a 1.50R target.
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


class Alpha20VWAPTrendContinuation(BaseHypothesis):
    """
    Intraday VWAP Momentum Breakout (Alpha 20):
    1. Daily Macro Trend: Prior Day Close > Prior Day Open (for Long) or Prior Day Close < Prior Day Open (for Short).
    2. Cumulative Session VWAP (09:15 onwards): sum(price * vol) / sum(vol).
    3. Trigger Window (10:00 to 11:30 IST):
       - Long: 15m Close crosses ABOVE VWAP + Bullish Candle + RVOL >= 1.30x (in Bullish Macro Trend).
       - Short: 15m Close crosses BELOW VWAP + Bearish Candle + RVOL >= 1.30x (in Bearish Macro Trend).
    4. Execution & Sizing: Next-bar open fill, Stop at VWAP - 0.20 * Daily ATR, Target = 1.50R, 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_20_VWAP_TREND_CONTINUATION",
            name="Alpha_20_VWAP_Trend_Continuation",
            category="INTRADAY_BENCHMARK_MOMENTUM",
            economic_rationale=(
                "When price crosses decisively across intraday VWAP with volume surge (RVOL >= 1.30x) "
                "in the direction of the daily macro trend, institutional VWAP execution algorithms "
                "drive persistent momentum follow-through."
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
            "min_rvol": 1.30,                  # Volume >= 1.30x shifted TOD baseline
            "target_rr": 1.50,                 # 1.50R target multiple
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.20, 1.30, 1.40],
            "target_rr": [1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 20.
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
        prev_prev_close = daily_summary["day_close"].shift(2)

        macro_trend = pd.Series(0, index=daily_summary.index)
        macro_trend[prev_close > prev_open] = 1
        macro_trend[prev_close < prev_open] = -1

        # Daily ATR(14) (Shifted 1 session)
        tr1 = prev_high - prev_low
        tr2 = (prev_high - prev_prev_close).abs()
        tr3 = (prev_low - prev_prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean()

        out["macro_trend"] = pd.Series(dates, index=out.index).map(macro_trend).ffill()
        out["daily_atr"] = pd.Series(dates, index=out.index).map(daily_atr14).ffill()

        # 2. Cumulative Intraday Session VWAP
        typical_price = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_price * out["volume"]
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_vol"] = out["volume"].groupby(dates).cumsum()
        out["vwap"] = (out["cum_pv"] / out["cum_vol"].replace(0, np.nan)).ffill()

        # 3. 20-Session TOD Rolling Volume Baseline (Shifted 1 session)
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
        volumes = out["volume"].values
        tod_vols = out["tod_mean_vol"].values
        vwaps = out["vwap"].values
        daily_atrs = out["daily_atr"].values
        macro_trends = out["macro_trend"].values

        min_rvol = float(self.parameters.get("min_rvol", 1.30))
        target_rr = float(self.parameters.get("target_rr", 1.50))

        current_day = None
        traded_today = False

        for i in range(1, n):
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            if bar_date != current_day:
                current_day = bar_date
                traded_today = False

            if traded_today:
                continue

            # Evaluate strictly in morning window (10:00 to 11:30)
            if (hour == 10) or (hour == 11 and minute <= 30):
                c_close = closes[i]
                c_open = opens[i]
                p_close = closes[i - 1]
                c_vol = volumes[i]
                c_tod = tod_vols[i]
                c_vwap = vwaps[i]
                p_vwap = vwaps[i - 1]
                c_atr = daily_atrs[i]
                c_trend = macro_trends[i]

                if pd.isna(c_vwap) or pd.isna(c_atr) or c_atr <= 0 or c_trend == 0:
                    continue

                rvol = c_vol / max(1.0, c_tod)

                # Bullish VWAP Cross Above
                if (c_trend == 1) and (p_close <= p_vwap) and (c_close > c_vwap) and (c_close > c_open) and (rvol >= min_rvol):
                    signals[i] = 1.0
                    sl_price = c_vwap - (0.20 * c_atr)
                    stop_dist = max(c_close - sl_price, 0.15 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 20 VWAP LONG: Cross Above VWAP={c_vwap:.1f} | RVOL={rvol:.2f}x | "
                        f"SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

                # Bearish VWAP Cross Below
                elif (c_trend == -1) and (p_close >= p_vwap) and (c_close < c_vwap) and (c_close < c_open) and (rvol >= min_rvol):
                    signals[i] = -1.0
                    sl_price = c_vwap + (0.20 * c_atr)
                    stop_dist = max(sl_price - c_close, 0.15 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 20 VWAP SHORT: Cross Below VWAP={c_vwap:.1f} | RVOL={rvol:.2f}x | "
                        f"SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
