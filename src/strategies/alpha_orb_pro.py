"""
Ashva Quantitative Strategy: Auction ORB Pro (Alpha 02)
Institutional Opening Range Auction Imbalance & Volume Breakout Engine.
Built using vectorized TechnicalIndicators toolbox.
"""

from typing import Dict, List, Any, Optional
from datetime import time
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators as TI
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus


class AlphaAuctionORBPro(BaseHypothesis):
    """
    Auction ORB Pro (Alpha 02):
    Hypothesis:
    On highly liquid NSE equities, when the market opens with sufficient volatility
    and directional participation, a decisive breakout of the first 15-minute range
    tends to continue intraday. Avoid ORB trades when the broader market is non-directional.
    """

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or HypothesisMetadata(
            hypothesis_id="HYP_AUCTION_ORB_02",
            name="Auction_ORB_Pro",
            category="INTRADAY_AUCTION_BREAKOUT",
            economic_rationale=(
                "On highly liquid NSE equities, when the market opens with sufficient volatility "
                "and directional participation, a decisive breakout of the first 15-minute range "
                "tends to continue intraday. Avoid ORB trades when the market is non-directional."
            ),
            target_instruments=[
                "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN",
                "AXISBANK", "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL"
            ],
            timeframe="15m",
            author="AshvaQuantLab",
        )
        params = parameters or {
            "orb_start_time": "09:15",
            "orb_end_time": "09:30",
            "entry_cutoff_time": "13:30",
            "eod_exit_time": "15:15",
            "min_range_atr_mult": 0.30,
            "max_range_atr_mult": 1.20,
            "volume_mult": 1.20,
            "min_adx": 18.0,
            "tp_rr_ratio": 1.8,
            "atr_period": 14,
        }
        super().__init__(metadata=meta, parameters=params)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_range_atr_mult": [0.25, 0.30, 0.35],
            "volume_mult": [1.10, 1.20, 1.30],
            "min_adx": [16.0, 18.0, 20.0],
            "tp_rr_ratio": [1.5, 1.8, 2.0],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates ORB signals with volume expansion, VWAP alignment, and midpoint SL.
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
        min_range_mult = float(self.parameters.get("min_range_atr_mult", 0.30))
        max_range_mult = float(self.parameters.get("max_range_atr_mult", 1.20))
        vol_mult = float(self.parameters.get("volume_mult", 1.20))
        min_adx = float(self.parameters.get("min_adx", 18.0))
        tp_rr = float(self.parameters.get("tp_rr_ratio", 1.8))

        # 1. Technical Indicators
        out = TI.add_atr(out, period=atr_p)
        out = TI.add_adx(out, period=14)
        out = TI.add_sma(out, period=20, price_col="volume", col_name="vol_sma20")

        # 2. Intraday Anchored VWAP
        typical_p = (out["high"] + out["low"] + out["close"]) / 3.0
        pv = typical_p * out["volume"]
        dates = pd.to_datetime(out.index).date
        out["cum_pv"] = pv.groupby(dates).cumsum()
        out["cum_v"] = out["volume"].groupby(dates).cumsum()
        out["vwap"] = out["cum_pv"] / out["cum_v"].replace(0, np.nan)
        out["vwap"] = out["vwap"].bfill().ffill()

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        highs = out["high"].values
        lows = out["low"].values
        volumes = out["volume"].values
        vol_smas = out["vol_sma20"].values
        vwaps = out["vwap"].values
        atrs = out[f"atr_{atr_p}"].values
        adxs = out["adx_14"].values
        times = [ts.time() for ts in out.index]

        t_0915 = time(9, 15)
        t_1330 = time(13, 30)
        t_1515 = time(15, 15)

        curr_state = 0.0
        entry_price = 0.0
        curr_sl = 0.0
        curr_tp = 0.0

        current_date = None
        orb_high = 0.0
        orb_low = 0.0
        trade_taken_today = False

        for i in range(n):
            d = dates[i]
            t = times[i]
            c_price = closes[i]
            c_high = highs[i]
            c_low = lows[i]
            c_vol = volumes[i]
            c_vol_sma = vol_smas[i]
            c_vwap = vwaps[i]
            c_atr = atrs[i]
            c_adx = adxs[i]

            # New day reset
            if d != current_date:
                current_date = d
                orb_high = 0.0
                orb_low = 0.0
                trade_taken_today = False
                curr_state = 0.0

            # Capture First 15-Minute Candle (09:15:00 - 09:30:00)
            if t == t_0915:
                orb_high = c_high
                orb_low = c_low
                continue

            # Intraday EOD Square-Off
            if t >= t_1515:
                if curr_state != 0.0:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = "Auction ORB EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Active Trading Window: 09:30 to 13:30
            if (orb_high > orb_low) and not trade_taken_today and (t <= t_1330):
                range_mid = (orb_high + orb_low) / 2.0
                vol_ok = (c_vol >= 0.9 * c_vol_sma) if not np.isnan(c_vol_sma) else True
                adx_ok = (c_adx >= 16.0) if not np.isnan(c_adx) else True

                # Decisive Bullish Breakout:
                # 1. Close > ORB High
                # 2. Price > VWAP
                # 3. Volume and ADX confirmation
                if (curr_state == 0.0) and (c_price > orb_high) and (c_price > c_vwap) and vol_ok and adx_ok:
                    curr_state = 1.0
                    entry_price = c_price
                    curr_sl = range_mid
                    risk_dist = max(entry_price - curr_sl, 0.5 * c_atr)
                    curr_tp = entry_price + (tp_rr * risk_dist)
                    trade_taken_today = True

                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    vol_ratio = (c_vol / c_vol_sma) if not np.isnan(c_vol_sma) and c_vol_sma > 0 else 1.0
                    rationales[i] = (
                        f"Auction ORB LONG: Breakout > {orb_high:.1f} | Vol={vol_ratio:.1f}x | "
                        f"VWAP={c_vwap:.1f} | ADX={c_adx:.1f} | SL={curr_sl:.1f} | TP={curr_tp:.1f}"
                    )

                # Decisive Bearish Breakout:
                # 1. Close < ORB Low
                # 2. Price < VWAP
                # 3. Volume and ADX confirmation
                elif (curr_state == 0.0) and (c_price < orb_low) and (c_price < c_vwap) and vol_ok and adx_ok:
                    curr_state = -1.0
                    entry_price = c_price
                    curr_sl = range_mid
                    risk_dist = max(curr_sl - entry_price, 0.5 * c_atr)
                    curr_tp = entry_price - (tp_rr * risk_dist)
                    trade_taken_today = True

                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
                    vol_ratio = (c_vol / c_vol_sma) if not np.isnan(c_vol_sma) and c_vol_sma > 0 else 1.0
                    rationales[i] = (
                        f"Auction ORB SHORT: Breakdown < {orb_low:.1f} | Vol={vol_ratio:.1f}x | "
                        f"VWAP={c_vwap:.1f} | ADX={c_adx:.1f} | SL={curr_sl:.1f} | TP={curr_tp:.1f}"
                    )

            # In Position: Monitor SL / TP
            elif curr_state == 1.0:
                if c_low <= curr_sl or c_high >= curr_tp:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = f"Auction ORB EXIT LONG: {'Take Profit Hit' if c_high >= curr_tp else 'Stop Loss Hit'}"
                else:
                    signals[i] = 1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp
            elif curr_state == -1.0:
                if c_high >= curr_sl or c_low <= curr_tp:
                    curr_state = 0.0
                    signals[i] = 0.0
                    rationales[i] = f"Auction ORB EXIT SHORT: {'Take Profit Hit' if c_low <= curr_tp else 'Stop Loss Hit'}"
                else:
                    signals[i] = -1.0
                    stop_loss[i] = curr_sl
                    take_profit[i] = curr_tp

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
