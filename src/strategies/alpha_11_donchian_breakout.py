"""
Ashva Quantitative Strategy: Multi-Day 20-Day Donchian Breakout (Alpha 11 - Swing Momentum)
Captures multi-day institutional trend continuation following a clean breakout of the 20-day Donchian High.

Hypothesis:
When a liquid equity in a verified macro uptrend (EMA20 > EMA50 and Daily ADX >= 25) breaks above its
20-day Donchian Channel High with volume expansion, institutional accumulation drives a multi-day
trending impulse.
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


class Alpha11DonchianBreakout(BaseHypothesis):
    """
    Multi-Day 20-Day Donchian Breakout (Alpha 11 - Swing Momentum):
    1. 20-Day Donchian High: max(High_20d) strictly from completed prior sessions.
    2. Trend Regime Filter: Daily EMA20 > Daily EMA50 and Daily ADX(14) >= 25.0.
    3. Breakout Trigger: Intraday Close > 20-day Donchian High with RVOL >= 1.20x.
    4. Trailing Swing Execution: Stop at Entry - 1.5 * Daily ATR, Target at Entry + 3.0 * Daily ATR (1:2.0 RR).
    5. Holding Duration: Multi-day carry up to 10 trading days (250 15m bars).
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_11_DONCHIAN_BREAKOUT",
            name="Alpha_11_Donchian_Breakout",
            category="TREND_FOLLOWING_SWING_BREAKOUT",
            economic_rationale=(
                "When a liquid equity in a verified macro uptrend (EMA20 > EMA50 and Daily ADX >= 25) "
                "breaks above its 20-day Donchian Channel High with volume expansion, institutional "
                "accumulation drives a multi-day trending impulse."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
                "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
                "BAJFINANCE", "MARUTI", "SUNPHARMA"
            ],
            timeframe="15m",
            horizon=StrategyHorizon.SWING,
            mechanism=MarketMechanism.BREAKOUT,
            author="AshvaQuantLab",
        )
        params = parameters or {
            "donchian_window_days": 20,        # 20-day Donchian channel window
            "min_daily_adx": 25.0,             # Trending regime gate (Daily ADX >= 25)
            "min_rvol": 1.20,                  # Breakout volume >= 1.20x TOD baseline
            "stop_atr_mult": 1.50,             # Stop Loss = Entry - 1.50 * Daily ATR
            "target_atr_mult": 3.00,           # Target = Entry + 3.00 * Daily ATR (1:2.0 RR)
            "max_hold_bars": 250,              # Max hold: ~10 trading days (250 15m bars)
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "donchian_window_days": [15, 20, 25],
            "min_daily_adx": [22.0, 25.0, 28.0],
            "stop_atr_mult": [1.20, 1.50, 1.80],
            "target_atr_mult": [2.50, 3.00, 3.50],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic, zero look-ahead signal generation for Alpha 11.
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

        win_days = int(self.parameters.get("donchian_window_days", 20))
        min_adx = float(self.parameters.get("min_daily_adx", 25.0))
        min_rvol = float(self.parameters.get("min_rvol", 1.20))
        sl_atr_mult = float(self.parameters.get("stop_atr_mult", 1.50))
        tp_atr_mult = float(self.parameters.get("target_atr_mult", 3.00))
        max_hold = int(self.parameters.get("max_hold_bars", 250))

        # 20-Day Donchian High (Shifted 1 session)
        daily_donchian_high = daily_summary["day_high"].rolling(win_days, min_periods=win_days).max().shift(1)

        # Daily ATR(14) (Shifted 1 session)
        prev_close = daily_summary["day_close"].shift(1)
        tr1 = daily_summary["day_high"] - daily_summary["day_low"]
        tr2 = (daily_summary["day_high"] - prev_close).abs()
        tr3 = (daily_summary["day_low"] - prev_close).abs()
        daily_tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr14 = daily_tr.rolling(14, min_periods=5).mean().shift(1)

        # Daily EMAs (20 & 50) and Daily ADX (Shifted 1 session)
        daily_ema20 = daily_summary["day_close"].ewm(span=20, adjust=False).mean().shift(1)
        daily_ema50 = daily_summary["day_close"].ewm(span=50, adjust=False).mean().shift(1)

        daily_adx_df = TI.add_adx(
            daily_summary.rename(columns={"day_high": "high", "day_low": "low", "day_close": "close"}),
            period=14
        )
        daily_adx14 = daily_adx_df["adx_14"].shift(1)

        # 20-Session TOD Rolling Volume Baseline (Shifted 1 session)
        tod_rolling = out.groupby("time_str")["volume"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(out["volume"])
        out["tod_mean_vol"] = tod_rolling

        # Map daily features to 15m bars
        out["donchian_high_20d"] = pd.Series(dates, index=out.index).map(daily_donchian_high).ffill()
        out["daily_atr"] = pd.Series(dates, index=out.index).map(daily_atr14).ffill()
        out["daily_ema20"] = pd.Series(dates, index=out.index).map(daily_ema20).ffill()
        out["daily_ema50"] = pd.Series(dates, index=out.index).map(daily_ema50).ffill()
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
        volumes = out["volume"].values
        tod_vols = out["tod_mean_vol"].values
        donch_highs = out["donchian_high_20d"].values
        daily_atrs = out["daily_atr"].values
        ema20s = out["daily_ema20"].values
        ema50s = out["daily_ema50"].values
        adxs = out["daily_adx"].values

        in_swing_trade = False
        bars_in_trade = 0
        active_sl = 0.0
        active_tp = 0.0

        for i in range(1, n):
            c_close = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_tod = tod_vols[i]
            c_dhigh = donch_highs[i]
            c_atr = daily_atrs[i]
            c_e20 = ema20s[i]
            c_e50 = ema50s[i]
            c_adx = adxs[i]

            # -------------------------------------------------------------
            # Manage Active Swing Position
            # -------------------------------------------------------------
            if in_swing_trade:
                bars_in_trade += 1

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
                        "Alpha 11 SWING EXIT: "
                        + ("Take Profit Hit" if hit_tp else ("Stop Loss Hit" if hit_sl else "Max 10-Day Duration Reached"))
                    )
                else:
                    signals[i] = 1.0  # Holding swing position across days
                    stop_loss[i] = active_sl
                    take_profit[i] = active_tp
                    rationales[i] = rationales[i - 1]
                continue

            # -------------------------------------------------------------
            # Evaluate New 20-Day Donchian Breakout Entry
            # -------------------------------------------------------------
            if pd.isna(c_dhigh) or pd.isna(c_atr) or pd.isna(c_adx) or pd.isna(c_e20) or pd.isna(c_e50):
                continue

            # 1. Macro Trend Alignment: Daily EMA20 > Daily EMA50
            if c_e20 <= c_e50:
                continue

            # 2. Trending Regime Filter: Daily ADX >= 25
            if c_adx < min_adx:
                continue

            # 3. 20-Day Breakout Trigger with Volume Expansion
            rvol = c_vol / max(1.0, c_tod)
            if (c_close > c_dhigh) and (c_close > c_open) and (rvol >= min_rvol):
                sl_price = c_close - (sl_atr_mult * c_atr)
                tp_price = c_close + (tp_atr_mult * c_atr)

                signals[i] = 1.0
                stop_loss[i] = sl_price
                take_profit[i] = tp_price
                active_sl = sl_price
                active_tp = tp_price
                in_swing_trade = True
                bars_in_trade = 0
                rationales[i] = (
                    f"Alpha 11 SWING LONG: 20D_High=Rs {c_dhigh:.1f} | ADX={c_adx:.1f} | "
                    f"RVOL={rvol:.2f}x | SL=Rs {sl_price:.1f} | TP=Rs {tp_price:.1f} (1:2.0 RR)"
                )

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
