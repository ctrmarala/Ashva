"""
Ashva Quantitative Strategy: 13:30 European Open Momentum Surge (Alpha 12 - Intraday Cross-Market Momentum)
Captures afternoon institutional trend expansion triggered by London / European market open liquidity (13:30 IST).

Hypothesis:
At 13:30 IST, European markets (London LSE & Frankfurt DAX) open, injecting major international institutional
order flow into liquid Indian blue chips (Banking, IT, Energy). When a stock breaks out of its 11:30-13:30 mid-day
consolidation box with high volume at 13:30/13:45, the momentum persists into the 15:15 close.
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


class Alpha12EuropeanOpenMomentum(BaseHypothesis):
    """
    13:30 European Open Momentum Surge (Alpha 12):
    1. Mid-Day Consolidation Box (11:30 to 13:15 IST): Computes High, Low, Range.
    2. European Open Window: 13:30 to 14:00 IST.
    3. Trigger: 15m Close breaks outside the mid-day box + same-direction candle body + RVOL >= 1.20x.
    4. Execution & Risk: Next-bar open fill, Stop at Mid-day Box Midpoint, 1:1.50 Target, Mandatory 15:15 EOD Exit.
    5. Intraday Cost Model: Segment.EQUITY_INTRADAY.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_12_EUROPEAN_OPEN_MOMENTUM",
            name="Alpha_12_European_Open_Momentum",
            category="CROSS_MARKET_AFTERNOON_MOMENTUM",
            economic_rationale=(
                "At 13:30 IST, European markets open, injecting major international institutional "
                "order flow into liquid Indian blue chips. When a stock breaks out of its 11:30-13:30 "
                "mid-day consolidation box with elevated volume at 13:30/13:45, the momentum tends to "
                "persist into the 15:15 afternoon close."
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
            "target_rr": 1.50,                 # Exactly 1:1.50 Risk-to-Reward ratio
            "max_box_atr_ratio": 0.60,         # Mid-day box range <= 0.60 * Daily ATR
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_rvol": [1.10, 1.20, 1.30],
            "target_rr": [1.50, 2.00],
            "max_box_atr_ratio": [0.50, 0.60, 0.70],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 12.
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

        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_box_atr = float(self.parameters.get("max_box_atr_ratio", 0.60))

        current_day = None
        box_high = 0.0
        box_low = 999999.0
        traded_today = False

        for i in range(n):
            ts = timestamps[i]
            bar_date = dates[i]
            bar_time = times[i]
            hour = bar_time.hour
            minute = bar_time.minute

            # Reset on new day
            if bar_date != current_day:
                current_day = bar_date
                box_high = 0.0
                box_low = 999999.0
                traded_today = False

            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_atr = daily_atrs[i]

            # -------------------------------------------------------------
            # 1. Track Mid-Day Consolidation Box (11:30 to 13:15 IST)
            # -------------------------------------------------------------
            if (hour == 11 and minute >= 30) or (hour == 12) or (hour == 13 and minute <= 15):
                box_high = max(box_high, c_high)
                box_low = min(box_low, c_low)
                continue

            # -------------------------------------------------------------
            # 2. European Open Expansion Trigger (13:30 to 14:00 IST)
            # -------------------------------------------------------------
            if traded_today or box_high <= 0 or box_low >= 999999.0 or pd.isna(c_atr) or c_atr <= 0:
                continue

            if (hour == 13 and minute in [30, 45]) or (hour == 14 and minute == 0):
                box_range = box_high - box_low
                if box_range <= 0.01 or box_range > (max_box_atr * c_atr):
                    continue

                box_mid = (box_high + box_low) / 2.0
                rvol = c_vol / max(1.0, c_tod)

                # Bullish European Open Breakout (LONG)
                if (c_close > box_high) and (c_close > c_open) and (rvol >= min_rvol):
                    signals[i] = 1.0
                    stop_dist = max(c_close - box_mid, 0.20 * c_atr)
                    stop_loss[i] = c_close - stop_dist
                    take_profit[i] = c_close + (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 12 LONG: European Open Breakout Close={c_close:.1f} > BoxHigh={box_high:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

                # Bearish European Open Breakdown (SHORT)
                elif (c_close < box_low) and (c_close < c_open) and (rvol >= min_rvol):
                    signals[i] = -1.0
                    stop_dist = max(box_mid - c_close, 0.20 * c_atr)
                    stop_loss[i] = c_close + stop_dist
                    take_profit[i] = c_close - (target_rr * stop_dist)
                    rationales[i] = (
                        f"Alpha 12 SHORT: European Open Breakdown Close={c_close:.1f} < BoxLow={box_low:.1f} | "
                        f"RVOL={rvol:.2f}x | SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f} (1:{target_rr:.1f} RR)"
                    )
                    traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
