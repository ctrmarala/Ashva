"""
Ashva Quantitative Strategy: Alpha 03 - VWAP Mean Reversion (alpha_03_vwap_reversion)
Institutional VWAP Statistical Deviation & Liquidity Exhaustion Mean Reversion Engine.
Built using vectorized TechnicalIndicators and Microstructure feature toolbox.
"""

from typing import Dict, List, Any, Optional
from datetime import time
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators as TI
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus


class Alpha03VWAPReversion(BaseHypothesis):
    """
    Alpha 03: VWAP Mean Reversion (alpha_03_vwap_reversion)
    Hypothesis:
    Highly liquid NSE stocks that make an unusually large intraday move away from VWAP
    tend to revert toward VWAP when the broader market is not strongly trending.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_ALPHA_03_VWAP_REVERSION",
            name="alpha_03_vwap_reversion",
            category="INTRADAY_VWAP_MEAN_REVERSION",
            economic_rationale=(
                "Highly liquid NSE stocks that make an unusually large intraday move away from VWAP "
                "tend to revert toward VWAP when the broader market is not strongly trending. "
                "Institutional execution algorithms seek fair value around VWAP, creating counter-trend liquidity."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN",
                "AXISBANK", "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL"
            ],
            timeframe="15m",
            author="AshvaQuantLab",
        )
        params = parameters or {
            "vwap_std_multiplier": 2.0,
            "max_adx_trend_cap": 25.0,  # Avoid trending runaway regimes
            "rsi_oversold": 32.0,
            "rsi_overbought": 68.0,
            "atr_period": 14,
            "sl_atr_multiplier": 1.2,
            "tp_target": "VWAP",        # Target reversion back to VWAP
            "entry_cutoff_time": "14:15",
            "eod_exit_time": "15:15",
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "vwap_std_multiplier": [1.8, 2.0, 2.2],
            "max_adx_trend_cap": [22.0, 25.0, 28.0],
            "rsi_oversold": [30.0, 32.0, 35.0],
            "rsi_overbought": [65.0, 68.0, 70.0],
            "sl_atr_multiplier": [1.0, 1.2, 1.5],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates mean-reversion signals when price stretches >= 2.0 std dev from VWAP in non-trending markets.
        """
        out = df.copy()

        # Ensure datetime index
        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or 'timestamp' column")

        atr_p = int(self.parameters.get("atr_period", 14))
        std_m = float(self.parameters.get("vwap_std_multiplier", 2.0))
        max_adx = float(self.parameters.get("max_adx_trend_cap", 25.0))
        rsi_os = float(self.parameters.get("rsi_oversold", 32.0))
        rsi_ob = float(self.parameters.get("rsi_overbought", 68.0))
        sl_m = float(self.parameters.get("sl_atr_multiplier", 1.2))

        # 1. Technical Indicators
        out = TI.add_atr(out, period=atr_p)
        out = TI.add_rsi(out, period=14, price_col="close")
        out = TI.add_adx(out, period=14)

        # 2. Intraday Anchored VWAP & Standard Deviation Dispersion Bands
        typical_p = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_p * out["volume"]
        price_sq_v = (typical_p ** 2) * out["volume"]
        dates = pd.to_datetime(out.index).date

        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_v"] = out["volume"].groupby(dates).cumsum()
        out["cum_price_sq_v"] = price_sq_v.groupby(dates).cumsum()

        out["vwap"] = out["cum_pv"] / out["cum_v"].replace(0, np.nan)
        out["vwap"] = out["vwap"].bfill().ffill()

        variance = (out["cum_price_sq_v"] / out["cum_v"]) - (out["vwap"] ** 2)
        variance = variance.clip(lower=0)
        out["vwap_std"] = np.sqrt(variance).bfill().ffill()

        out["vwap_upper"] = out["vwap"] + (std_m * out["vwap_std"])
        out["vwap_lower"] = out["vwap"] - (std_m * out["vwap_std"])

        # Clean intermediate columns
        out.drop(columns=["cum_pv", "cum_v", "cum_price_sq_v"], inplace=True)

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        opens = out["open"].values
        highs = out["high"].values
        lows = out["low"].values
        vwaps = out["vwap"].values
        vwap_uppers = out["vwap_upper"].values
        vwap_lowers = out["vwap_lower"].values
        rsis = out["rsi_14"].values
        atrs = out[f"atr_{atr_p}"].values
        adxs = out["adx_14"].values
        times = [ts.time() for ts in out.index]

        t_0930 = time(9, 30)
        t_1415 = time(14, 15)
        t_1515 = time(15, 15)

        curr_state = 0.0
        entry_price = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        last_trade_bar = -999

        for i in range(max(atr_p, 20), n):
            t = times[i]
            c_price = closes[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vwap = vwaps[i]
            c_upper = vwap_uppers[i]
            c_lower = vwap_lowers[i]
            c_rsi = rsis[i]
            c_atr = atrs[i]
            c_adx = adxs[i]

            # Intraday EOD Square-Off
            if t >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "alpha_03_vwap_reversion EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Trade Cooldown: at least 4 bars (1 hour)
            cooldown_ok = (i - last_trade_bar) >= 4
            non_trending = (c_adx <= max_adx) if not np.isnan(c_adx) else True

            # Entry Window: 09:30 to 14:15
            if (curr_state == 0.0) and cooldown_ok and non_trending and (t_0930 <= t <= t_1415):
                # Long Mean Reversion (Over-extended to downside):
                # 1. Low touched/pierced Lower VWAP Band (<= VWAP - 2.0 sigma)
                # 2. Bullish Rejection: Close > Open (buyer absorption)
                # 3. RSI <= 35 (Oversold condition)
                # 4. Market is not trending (ADX <= 25)
                if (c_low <= c_lower) and (c_price > c_open) and (c_rsi <= rsi_os):
                    curr_state = 1.0
                    entry_price = c_price
                    curr_sl = c_low - (sl_m * c_atr)
                    curr_tp = c_vwap  # Target reversion back to fair value VWAP
                    last_trade_bar = i

                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = (
                        f"alpha_03_vwap_reversion LONG: Oversold at Lower Band ({c_lower:.1f}) | "
                        f"RSI={c_rsi:.1f} | ADX={c_adx:.1f} (Non-Trending) | Target VWAP={c_vwap:.1f} | SL={curr_sl:.1f}"
                    )

                # Short Mean Reversion (Over-extended to upside):
                # 1. High touched/pierced Upper VWAP Band (>= VWAP + 2.0 sigma)
                # 2. Bearish Rejection: Close < Open (seller absorption)
                # 3. RSI >= 65 (Overbought condition)
                # 4. Market is not trending (ADX <= 25)
                elif (c_high >= c_upper) and (c_price < c_open) and (c_rsi >= rsi_ob):
                    curr_state = -1.0
                    entry_price = c_price
                    curr_sl = c_high + (sl_m * c_atr)
                    curr_tp = c_vwap  # Target reversion back to fair value VWAP
                    last_trade_bar = i

                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    rationales[i] = (
                        f"alpha_03_vwap_reversion SHORT: Overbought at Upper Band ({c_upper:.1f}) | "
                        f"RSI={c_rsi:.1f} | ADX={c_adx:.1f} (Non-Trending) | Target VWAP={c_vwap:.1f} | SL={curr_sl:.1f}"
                    )

            # In Position: Monitor Stop Loss & VWAP Mean Reversion Take Profit
            elif curr_state == 1.0:
                # Exit when price touches or crosses above VWAP, or hits Stop Loss
                if c_high >= c_vwap or c_low <= curr_sl:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = f"alpha_03_vwap_reversion EXIT LONG: {'VWAP Reversion Target Hit' if c_high >= c_vwap else 'Stop Loss Hit'}"
                else:
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = c_vwap  # Dynamic VWAP target
            elif curr_state == -1.0:
                # Exit when price touches or crosses below VWAP, or hits Stop Loss
                if c_low <= c_vwap or c_high >= curr_sl:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = f"alpha_03_vwap_reversion EXIT SHORT: {'VWAP Reversion Target Hit' if c_low <= c_vwap else 'Stop Loss Hit'}"
                else:
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = c_vwap  # Dynamic VWAP target

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
