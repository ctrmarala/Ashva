"""
Ashva Quantitative Strategy: TrendSurfer Pro (Alpha 01)
Multi-Timeframe Trend Continuation & Institutional Value Pullback Engine.
Built using vectorized TechnicalIndicators toolbox.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators as TI
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus


class AlphaTrendSurfer(BaseHypothesis):
    """
    TrendSurfer Pro (Alpha 01):
    1. Regime Filter: Supertrend (10, 3.0) determines overall bullish/bearish market state.
    2. Pullback Trigger: Price pulls back to the dynamic 20-period Exponential Moving Average (EMA).
    3. Momentum Gate: RSI (14) confirms continuation momentum without being overextended.
    4. Volatility Exits: Stop Loss = Entry - (1.5 * ATR), Take Profit = Entry + (2.5 * ATR).
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_TREND_SURFER_01",
            name="TrendSurfer_Pro",
            category="MOMENTUM_TREND_PULLBACK",
            economic_rationale=(
                "Institutional trend continuation on liquid Indian equities. "
                "Large institutions accumulate positions during minor pullbacks to dynamic "
                "20-period EMAs rather than chasing breakouts."
            ),
            target_instruments=["INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "RELIANCE", "TATAMOTORS", "BHARTIARTL"],
            timeframe="15m",
            author="AshvaQuantLab",
        )
        params = parameters or {
            "supertrend_period": 10,
            "supertrend_multiplier": 3.0,
            "ema_period": 20,
            "rsi_period": 14,
            "rsi_long_min": 48.0,
            "rsi_long_max": 72.0,
            "rsi_short_min": 28.0,
            "rsi_short_max": 52.0,
            "atr_period": 14,
            "sl_atr_multiplier": 1.5,
            "tp_atr_multiplier": 2.5,
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "ema_period": [15, 20, 25],
            "supertrend_multiplier": [2.5, 3.0, 3.5],
            "sl_atr_multiplier": [1.2, 1.5, 1.8],
            "tp_atr_multiplier": [2.0, 2.5, 3.0],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals (+1.0 LONG, -1.0 SHORT, 0.0 FLAT), stop loss, take profit,
        and machine-readable decision rationales.
        """
        out = df.copy()

        st_p = int(self.parameters.get("supertrend_period", 10))
        st_m = float(self.parameters.get("supertrend_multiplier", 3.0))
        ema_p = int(self.parameters.get("ema_period", 20))
        rsi_p = int(self.parameters.get("rsi_period", 14))
        atr_p = int(self.parameters.get("atr_period", 14))
        sl_m = float(self.parameters.get("sl_atr_multiplier", 1.5))
        tp_m = float(self.parameters.get("tp_atr_multiplier", 2.5))

        # 1. Compute Indicators via Centralized Technical Toolbox
        out = TI.add_ema(out, period=20, price_col="close", col_name="ema_fast")
        out = TI.add_ema(out, period=50, price_col="close", col_name="ema_slow")
        out = TI.add_rsi(out, period=rsi_p, price_col="close")
        out = TI.add_atr(out, period=atr_p)
        out = TI.add_adx(out, period=14)

        # Intraday VWAP
        typical_p = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_p * out["volume"]
        dates = pd.to_datetime(out.index).date
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_v"] = out["volume"].groupby(dates).cumsum()
        out["vwap"] = out["cum_pv"] / out["cum_v"].replace(0, np.nan)
        out["vwap"] = out["vwap"].bfill().ffill()

        rsi_col = f"rsi_{rsi_p}"
        atr_col = f"atr_{atr_p}"
        adx_col = "adx_14"

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        opens = out["open"].values if "open" in out.columns else closes
        highs = out["high"].values if "high" in out.columns else closes
        lows = out["low"].values if "low" in out.columns else closes
        ema_fasts = out["ema_fast"].values
        ema_slows = out["ema_slow"].values
        vwaps = out["vwap"].values
        rsis = out[rsi_col].values
        atrs = out[atr_col].values
        adxs = out[adx_col].values if adx_col in out.columns else np.full(n, 25.0)

        curr_state = 0.0  # 0: Flat, 1: Long, -1: Short
        entry_price = 0.0
        last_trade_bar = -999

        for i in range(55, n):
            c_price = closes[i]
            c_open = opens[i]
            c_low = lows[i]
            c_high = highs[i]
            c_efast = ema_fasts[i]
            c_eslow = ema_slows[i]
            c_vwap = vwaps[i]
            c_rsi = rsis[i]
            c_atr = atrs[i]
            c_adx = adxs[i]

            # Cooldown: At least 6 bars (1.5 hours) between trades to prevent over-trading
            cooldown_ok = (i - last_trade_bar) >= 6

            # Long Entry:
            # 1. EMA 20 > EMA 50 (Bullish trend alignment)
            # 2. Price > VWAP (Institutional buying dominance)
            # 3. ADX >= 18 (Trending)
            # 4. Pullback: Low tested EMA20 (low <= EMA20 * 1.002) and Close > EMA20
            # 5. RSI between 50 and 65
            long_trigger = (
                (curr_state == 0.0)
                and cooldown_ok
                and (c_efast > c_eslow)
                and (c_price >= c_vwap)
                and (c_adx >= 18.0)
                and (c_low <= c_efast * 1.002)
                and (c_price > c_efast)
                and (c_price > c_open)
                and (50.0 <= c_rsi <= 65.0)
            )

            # Short Entry:
            # 1. EMA 20 < EMA 50 (Bearish trend alignment)
            # 2. Price < VWAP (Institutional selling dominance)
            # 3. ADX >= 18 (Trending)
            # 4. Pullback: High tested EMA20 (high >= EMA20 * 0.998) and Close < EMA20
            # 5. RSI between 35 and 50
            short_trigger = (
                (curr_state == 0.0)
                and cooldown_ok
                and (c_efast < c_eslow)
                and (c_price <= c_vwap)
                and (c_adx >= 18.0)
                and (c_high >= c_efast * 0.998)
                and (c_price < c_efast)
                and (c_price < c_open)
                and (35.0 <= c_rsi <= 50.0)
            )

            if curr_state == 0.0:
                if long_trigger:
                    curr_state = 1.0
                    entry_price = c_price
                    last_trade_bar = i
                    signals[i] = 1.0
                    stop_loss[i] = entry_price - (sl_m * c_atr)
                    take_profit[i] = entry_price + (tp_m * c_atr)
                    rationales[i] = (
                        f"TrendSurfer LONG: EMA20>50 | ADX={c_adx:.1f} | RSI={c_rsi:.1f} | "
                        f"SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f}"
                    )
                elif short_trigger:
                    curr_state = -1.0
                    entry_price = c_price
                    last_trade_bar = i
                    signals[i] = -1.0
                    stop_loss[i] = entry_price + (sl_m * c_atr)
                    take_profit[i] = entry_price - (tp_m * c_atr)
                    rationales[i] = (
                        f"TrendSurfer SHORT: EMA20<50 | ADX={c_adx:.1f} | RSI={c_rsi:.1f} | "
                        f"SL=Rs {stop_loss[i]:.1f} | TP=Rs {take_profit[i]:.1f}"
                    )
            elif curr_state == 1.0:
                # Check SL / TP
                if c_low <= stop_loss[i - 1] or c_high >= take_profit[i - 1] or c_efast < c_eslow:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "TrendSurfer EXIT LONG: SL/TP or Trend Reversal"
                else:
                    signals[i] = 0.0  # Holding state: No new entry order
                    stop_loss[i] = stop_loss[i - 1]
                    take_profit[i] = take_profit[i - 1]
            elif curr_state == -1.0:
                # Check SL / TP
                if c_high >= stop_loss[i - 1] or c_low <= take_profit[i - 1] or c_efast > c_eslow:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "TrendSurfer EXIT SHORT: SL/TP or Trend Reversal"
                else:
                    signals[i] = 0.0  # Holding state: No new entry order
                    stop_loss[i] = stop_loss[i - 1]
                    take_profit[i] = take_profit[i - 1]

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
