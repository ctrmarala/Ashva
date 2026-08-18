"""
Ashva Quantitative Alpha 25: Cross-Sectional Residual Reversion
Hypothesis:
    When an individual stock makes an unusually large intraday move relative to the
    contemporaneous movement of the broader market while the broader market remains
    relatively calm, the stock's idiosyncratic residual move tends to mean-revert
    due to liquidity replenishment and fading of temporary order flow imbalances.

Mechanism:
    Cross-sectional residual mean reversion. Fades idiosyncratic outliers during calm market regimes.
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
from src.data.data_lake import DataLake


class Alpha25CrossSectionalResidualReversion(BaseHypothesis):
    """
    Alpha 25: Cross-Sectional Residual Mean Reversion Strategy.
    """

    # Class-level cache for contemporaneous universe median returns
    _universe_returns_cache: Dict[str, pd.DataFrame] = {}

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        meta = HypothesisMetadata(
            hypothesis_id="alpha_25",
            name="ALPHA_25_CROSS_SECTIONAL_RESIDUAL_REVERSION",
            category="CROSS_SECTIONAL_MEAN_REVERSION",
            economic_rationale=(
                "When an individual stock makes an unusually large intraday move relative to the "
                "contemporaneous movement of the broader market while the broader market is calm, "
                "the stock's idiosyncratic move is often driven by temporary liquidity imbalances "
                "rather than fundamental re-pricing, leading to statistical mean reversion."
            ),
            target_instruments=["NIFTY50_LIQUID"],
            timeframe="15m",
            horizon=StrategyHorizon.INTRADAY,
            mechanism=MarketMechanism.MEAN_REVERSION,
        )
        super().__init__(meta, parameters)

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "lookback_bars": [2, 4, 6],
            "min_residual_ratio_atr": [0.60, 0.75, 1.00],
            "max_market_move_pct": [0.0020, 0.0030, 0.0040],
            "target_rr": [1.25, 1.50, 2.00],
        }

    @classmethod
    def _get_universe_panel_returns(cls, lookback_bars: int) -> pd.DataFrame:
        """
        Loads and computes rolling returns for the universe to calculate contemporaneous median.
        Cached in class variable for fast execution across backtest assets.
        """
        cache_key = f"lookback_{lookback_bars}"
        if cache_key in cls._universe_returns_cache:
            return cls._universe_returns_cache[cache_key]

        lake = DataLake(read_only=True)
        universe = [
            "INFY", "TCS", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK",
            "KOTAKBANK", "RELIANCE", "LT", "TATASTEEL", "BHARTIARTL",
            "BAJFINANCE", "MARUTI", "SUNPHARMA"
        ]

        stock_returns = {}
        for sym in universe:
            df = lake.load_bars(sym, "15m")
            if df.empty:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # Rolling return over lookback_bars
            ret = (df["close"] - df["close"].shift(lookback_bars)) / df["close"].shift(lookback_bars).replace(0, np.nan)
            stock_returns[sym] = ret

        if stock_returns:
            panel = pd.DataFrame(stock_returns)
            cls._universe_returns_cache[cache_key] = panel
        else:
            cls._universe_returns_cache[cache_key] = pd.DataFrame()

        return cls._universe_returns_cache[cache_key]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates dense directional signals (+1.0 LONG, -1.0 SHORT, 0.0 FLAT/EXIT).
        """
        out = df.copy()

        if not isinstance(out.index, pd.DatetimeIndex):
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out.set_index("timestamp", inplace=True)
            else:
                out.index = pd.to_datetime(out.index)

        dates = out.index.date
        times = out.index.time

        # 1. Compute Daily ATR (14-day) anchored to prior days
        daily_df = out.resample("D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(daily_df) >= 14:
            daily_atr_df = TechnicalIndicators.add_atr(daily_df, period=14)
            daily_atr_prev = daily_atr_df["atr_14"].shift(1)
            atr_map = daily_atr_prev.to_dict()
            out["daily_atr"] = [atr_map.get(pd.Timestamp(d), np.nan) for d in dates]
        else:
            out["daily_atr"] = (out["high"] - out["low"]).rolling(14).mean()

        out["daily_atr"] = out["daily_atr"].ffill().bfill()

        # Strategy Hyperparameters
        lookback = int(self.parameters.get("lookback_bars", 4))
        min_res_ratio = float(self.parameters.get("min_residual_ratio_atr", 0.75))
        max_mkt_move = float(self.parameters.get("max_market_move_pct", 0.0030))
        target_rr = float(self.parameters.get("target_rr", 1.50))
        max_entry_hour = int(self.parameters.get("max_entry_hour", 14))

        # 2. Retrieve Cross-Sectional Universe Median Contemporaneous Returns
        panel_df = self._get_universe_panel_returns(lookback)
        if not panel_df.empty:
            mkt_median_series = panel_df.median(axis=1)
            out["mkt_return"] = mkt_median_series.reindex(out.index).fillna(0.0)
        else:
            out["mkt_return"] = 0.0

        # Compute stock rolling return over lookback_bars
        out["stock_return"] = (out["close"] - out["close"].shift(lookback)) / out["close"].shift(lookback).replace(0, np.nan)
        out["residual_return"] = out["stock_return"] - out["mkt_return"]

        n = len(out)
        signals = np.zeros(n, dtype=np.float64)
        stop_loss = np.zeros(n, dtype=np.float64)
        take_profit = np.zeros(n, dtype=np.float64)
        rationales = [""] * n

        closes = out["close"].values
        highs = out["high"].values
        lows = out["low"].values
        daily_atrs = out["daily_atr"].values
        stock_rets = out["stock_return"].values
        mkt_rets = out["mkt_return"].values
        residuals = out["residual_return"].values

        current_day = None
        traded_today = False
        curr_state = 0.0
        curr_sl = 0.0
        curr_tp = 0.0
        curr_rationale = ""

        t_1515 = pd.to_datetime("15:15:00").time()

        for i in range(lookback, n):
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
                    rationales[i] = "Alpha 25 EXIT: Intraday 15:15 EOD Square-Off"
                continue

            # Maintain active position across holding bars
            if curr_state != 0.0:
                signals[i] = curr_state
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                continue

            if traded_today or bar_time.hour > max_entry_hour:
                continue

            # Ensure all lookback bars belong to the SAME day
            prior_dates = dates[i - lookback : i + 1]
            if not all(d == bar_date for d in prior_dates):
                continue

            c_close = closes[i]
            c_high = highs[i]
            c_low = lows[i]
            c_atr = daily_atrs[i]
            c_sret = stock_rets[i]
            c_mret = mkt_rets[i]
            c_res = residuals[i]

            if pd.isna(c_atr) or c_atr <= 0 or pd.isna(c_res) or pd.isna(c_mret):
                continue

            # Check Market Calmness Gate
            if abs(c_mret) > max_mkt_move:
                continue

            # Required Idiosyncratic Residual Threshold (in price points)
            res_pts = abs(c_res) * c_close
            min_pts = min_res_ratio * c_atr

            if res_pts < min_pts:
                continue

            # -------------------------------------------------------------
            # Evaluate Mean-Reversion Triggers
            # -------------------------------------------------------------
            # Case 1: Idiosyncratic Over-Extension Upwards -> Enter SHORT (Fade Surge)
            if c_res > 0 and c_sret > 0:
                curr_state = -1.0
                stop_dist = max((c_high - c_close) + 0.15 * c_atr, 0.25 * c_atr)
                curr_sl = c_close + stop_dist
                curr_tp = c_close - (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 25 SHORT FADE: StockRet={c_sret*100:+.2f}% vs MktRet={c_mret*100:+.2f}% | "
                    f"Residual={c_res*100:+.2f}% ({res_pts:.1f} pts >= {min_pts:.1f} ATR) | "
                    f"SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                )
                signals[i] = -1.0
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                traded_today = True

            # Case 2: Idiosyncratic Over-Extension Downwards -> Enter LONG (Fade Plunge)
            elif c_res < 0 and c_sret < 0:
                curr_state = 1.0
                stop_dist = max((c_close - c_low) + 0.15 * c_atr, 0.25 * c_atr)
                curr_sl = c_close - stop_dist
                curr_tp = c_close + (target_rr * stop_dist)
                curr_rationale = (
                    f"Alpha 25 LONG FADE: StockRet={c_sret*100:+.2f}% vs MktRet={c_mret*100:+.2f}% | "
                    f"Residual={c_res*100:+.2f}% ({res_pts:.1f} pts >= {min_pts:.1f} ATR) | "
                    f"SL=Rs {curr_sl:.1f} | TP=Rs {curr_tp:.1f} (1:{target_rr:.1f} RR)"
                )
                signals[i] = 1.0
                stop_loss[i] = curr_sl
                take_profit[i] = curr_tp
                rationales[i] = curr_rationale
                traded_today = True

        out["signal"] = signals
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["rationale"] = rationales
        return out
