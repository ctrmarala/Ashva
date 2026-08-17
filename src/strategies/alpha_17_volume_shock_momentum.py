"""
Ashva Quantitative Strategy: Pre-Market Volume Shock Opening Momentum (Alpha 17)
Captures institutional block deal & opening auction volume surge momentum.

Hypothesis:
When Bar 1 (09:15-09:30) exhibits an extreme volume shock (RVOL >= 2.0x the 20-session baseline) with strong
directional candle body (>= 60% of candle range), major institutional portfolio execution is underway.
Entering on Bar 2 open with a stop at Bar 1 Midpoint captures directional momentum to a 1.50R target.
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


class Alpha17VolumeShockMomentum(BaseHypothesis):
    """
    Pre-Market Volume Shock Opening Momentum (Alpha 17):
    1. Extreme Volume Shock: Bar 1 (09:15) RVOL >= 2.0x 20-session baseline.
    2. Strong Directional Body: Body / Range >= 60%.
    3. Execution: Entry on 09:30 Open, Stop at Bar 1 Midpoint, Target = 1.50R, 15:15 EOD Exit.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_17_VOLUME_SHOCK_MOMENTUM",
            name="Alpha_17_Volume_Shock_Momentum",
            category="INSTITUTIONAL_VOLUME_SURGE",
            economic_rationale=(
                "When the opening 15-minute bar experiences an extreme volume shock (RVOL >= 2.0x) "
                "with strong directional candle body (>= 60%), institutional block executions drive "
                "intraday directional price momentum."
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
            "min_shock_rvol": 2.00,            # 2.0x minimum opening volume shock
            "min_body_ratio": 0.60,            # 60% minimum candle body ratio
            "target_rr": 1.50,                 # 1.50R target multiple
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_shock_rvol": [1.75, 2.00, 2.25],
            "min_body_ratio": [0.55, 0.60, 0.65],
            "target_rr": [1.50, 2.00],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 17.
        """
        out = df.copy()

        timestamps = pd.to_datetime(out.index)
        dates = timestamps.date
        times = timestamps.time
        out["time_str"] = [t.strftime("%H:%M") for t in times]

        # 1. Daily ATR(14) (Shifted 1 session)
        daily_summary = out.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last")
        )
        prev_close = daily_summary["day_close"].shift(1)
        tr1 = daily_summary["day_high"] - daily_summary["day_low"]
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)

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

        min_rvol = float(self.parameters.get("min_shock_rvol", 2.00))
        min_body = float(self.parameters.get("min_body_ratio", 0.60))
        target_rr = float(self.parameters.get("target_rr", 1.50))

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

            # Evaluate strictly on Bar 1 (09:15)
            if hour == 9 and minute == 15:
                c_close = closes[i]
                c_open = opens[i]
                c_high = highs[i]
                c_low = lows[i]
                c_vol = volumes[i]
                c_tod = tod_vols[i]
                c_atr = daily_atrs[i]

                if pd.isna(c_atr) or c_atr <= 0:
                    continue

                rvol = c_vol / max(1.0, c_tod)
                candle_range = max(c_high - c_low, 0.01)
                body_ratio = abs(c_close - c_open) / candle_range

                # Bullish Volume Shock
                if (rvol >= min_rvol) and (c_close > c_open) and (body_ratio >= min_body):
                    signals[i] = 1.0
                    bar1_mid = (c_high + c_low) / 2.0
                    stop_dist = max(c_close - bar1_mid, 0.15 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 17 SHOCK LONG: RVOL={rvol:.2f}x >= {min_rvol}x | Body={body_ratio*100:.1f}% | "
                        f"SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

                # Bearish Volume Shock
                elif (rvol >= min_rvol) and (c_close < c_open) and (body_ratio >= min_body):
                    signals[i] = -1.0
                    bar1_mid = (c_high + c_low) / 2.0
                    stop_dist = max(bar1_mid - c_close, 0.15 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 17 SHOCK SHORT: RVOL={rvol:.2f}x >= {min_rvol}x | Body={body_ratio*100:.1f}% | "
                        f"SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
