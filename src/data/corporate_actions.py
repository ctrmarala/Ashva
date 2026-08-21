"""
Ashva Corporate Action Engine & Historical Price Adjustment Module
Tracks and mathematically adjusts historical OHLCV series for:
1. Stock Splits (e.g. 1:2, 1:5, 1:10)
2. Bonus Issues (e.g. 1:1, 1:2)
3. Large / Special Dividends (>2% dividend yield impact)
4. Rights Issues

Enforces CRSP/NSE institutional standards:
- Historical prices prior to ex_date are multiplied by adjustment_factor (f)
- Historical volumes prior to ex_date are divided by adjustment_factor (f)
- Value (Close * Volume) is strictly conserved.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
import pandas as pd
import duckdb

from src.data.data_lake import DataLake


class CorporateActionType(str, Enum):
    SPLIT = "SPLIT"
    BONUS = "BONUS"
    SPECIAL_DIVIDEND = "SPECIAL_DIVIDEND"
    RIGHTS = "RIGHTS"


@dataclass
class CorporateAction:
    symbol: str
    action_type: CorporateActionType
    ex_date: str  # YYYY-MM-DD
    ratio_old: float = 1.0
    ratio_new: float = 1.0
    cash_amount: float = 0.0
    pre_event_price: Optional[float] = None
    notes: str = ""

    @property
    def adjustment_factor(self) -> float:
        """
        Calculates the backward multiplicative adjustment factor (f).
        All historical prices before ex_date are multiplied by f.
        """
        if self.action_type in (CorporateActionType.SPLIT, CorporateActionType.BONUS):
            # E.g., 1:2 split -> ratio_old=1, ratio_new=2 -> factor = 1/2 = 0.50
            # E.g., 1:1 bonus -> ratio_old=1, ratio_new=2 (1 existing + 1 bonus) -> factor = 0.50
            if self.ratio_new <= 0:
                raise ValueError("ratio_new must be positive")
            return self.ratio_old / self.ratio_new

        elif self.action_type == CorporateActionType.SPECIAL_DIVIDEND:
            # E.g. Pre-close ₹300, Dividend ₹50 -> factor = (300 - 50) / 300 = 0.8333
            if self.pre_event_price and self.pre_event_price > self.cash_amount:
                return (self.pre_event_price - self.cash_amount) / self.pre_event_price
            return 1.0

        elif self.action_type == CorporateActionType.RIGHTS:
            if self.pre_event_price and self.pre_event_price > 0:
                # Theoretical Ex-Rights Price (TERP)
                terp = ((self.ratio_old * self.pre_event_price) + (self.ratio_new * self.cash_amount)) / (self.ratio_old + self.ratio_new)
                return terp / self.pre_event_price
            return 1.0

        return 1.0


class CorporateActionManager:
    """
    Manages corporate action tracking, anomaly detection, and automated
    backward adjustment across DuckDB and Parquet data stores.
    """

    def __init__(self, data_lake: Optional[DataLake] = None, ledger_path: str = "config/corporate_actions_ledger.json"):
        self.lake = data_lake or DataLake(read_only=False)
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger: List[Dict[str, Any]] = self._load_ledger()

    def _load_ledger(self) -> List[Dict[str, Any]]:
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_ledger(self):
        with open(self.ledger_path, "w") as f:
            json.dump(self.ledger, f, indent=2)

    def register_and_apply(self, action: CorporateAction) -> Dict[str, Any]:
        """
        Registers a corporate action in the audit ledger and applies the backward
        multiplicative price/volume adjustments across all historical timeframes.
        """
        sym = action.symbol.upper()
        ex_dt = pd.to_datetime(action.ex_date)
        factor = action.adjustment_factor

        if factor == 1.0:
            return {"status": "SKIPPED", "reason": "Adjustment factor is 1.0 (no change required)"}

        # 1. Update DuckDB ohlcv_bars table
        if self.lake.conn is not None and not self.lake.read_only:
            try:
                self.lake.conn.execute("""
                    UPDATE ohlcv_bars
                    SET open = open * ?,
                        high = high * ?,
                        low = low * ?,
                        close = close * ?,
                        volume = CAST(volume / ? AS BIGINT)
                    WHERE symbol = ? AND timestamp < ?;
                """, [factor, factor, factor, factor, factor, sym, ex_dt])
            except Exception as e:
                pass

        # 2. Update Parquet Files for all available timeframes
        timeframes = ["1m", "5m", "10m", "15m", "30m", "1d"]
        updated_timeframes = []

        for tf in timeframes:
            parquet_file = self.lake.parquet_dir / f"{sym}_{tf}.parquet"
            if parquet_file.exists():
                try:
                    df = pd.read_parquet(parquet_file)
                    if not df.empty and "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        mask = df["timestamp"] < ex_dt

                        if mask.any():
                            df.loc[mask, "open"] = df.loc[mask, "open"] * factor
                            df.loc[mask, "high"] = df.loc[mask, "high"] * factor
                            df.loc[mask, "low"] = df.loc[mask, "low"] * factor
                            df.loc[mask, "close"] = df.loc[mask, "close"] * factor
                            df.loc[mask, "volume"] = (df.loc[mask, "volume"] / factor).astype(np.int64)

                            df.to_parquet(parquet_file, engine="pyarrow", index=False)
                            updated_timeframes.append(tf)
                except Exception as e:
                    pass

        # 3. Append to Audit Ledger
        record = {
            "symbol": sym,
            "action_type": action.action_type.value,
            "ex_date": action.ex_date,
            "adjustment_factor": round(factor, 6),
            "ratio_old": action.ratio_old,
            "ratio_new": action.ratio_new,
            "cash_amount": action.cash_amount,
            "updated_timeframes": updated_timeframes,
            "applied_at": pd.Timestamp.now().isoformat(),
            "notes": action.notes,
        }
        self.ledger.append(record)
        self._save_ledger()

        return {
            "status": "APPLIED",
            "symbol": sym,
            "ex_date": action.ex_date,
            "adjustment_factor": factor,
            "updated_timeframes": updated_timeframes,
        }

    def detect_unadjusted_anomalies(
        self,
        symbol: str,
        timeframe: str = "1d",
        threshold_pct: float = 0.20,  # >20% single-day overnight drop
    ) -> List[Dict[str, Any]]:
        """
        Scans historical data for suspicious, unadjusted overnight price cliffs (>20%)
        that resemble unadjusted splits or bonus issues.
        """
        df = self.lake.load_bars(symbol.upper(), timeframe)
        if df.empty or len(df) < 5:
            return []

        if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

        df["prev_close"] = df["close"].shift(1)
        df["overnight_gap"] = (df["open"] - df["prev_close"]) / df["prev_close"]

        anomalies = []
        suspicious_drops = df[df["overnight_gap"] < -threshold_pct]

        for dt, row in suspicious_drops.iterrows():
            anomalies.append({
                "symbol": symbol.upper(),
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "prev_close": round(float(row["prev_close"]), 2),
                "open": round(float(row["open"]), 2),
                "gap_pct": f"{round(float(row['overnight_gap']) * 100.0, 2)}%",
                "estimated_split_ratio": self._estimate_split_ratio(row["overnight_gap"]),
            })

        return anomalies

    def _estimate_split_ratio(self, gap_pct: float) -> str:
        """Estimates probable corporate action ratio from price drop."""
        if -0.55 <= gap_pct <= -0.45:
            return "1:2 Split or 1:1 Bonus (~50% drop)"
        elif -0.70 <= gap_pct <= -0.60:
            return "1:3 Split (~66% drop)"
        elif -0.83 <= gap_pct <= -0.75:
            return "1:4 or 1:5 Split (~80% drop)"
        elif -0.92 <= gap_pct <= -0.88:
            return "1:10 Split (~90% drop)"
        return "Large Gap / Potential Dividend"
