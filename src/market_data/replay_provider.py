"""
Ashva Replay Market Data Provider
Streams historical bars from DataLake in strict multi-symbol chronological synchronization.
Emulates live bar streaming for the Replay execution mode.
"""

from datetime import datetime
from typing import Dict, List, Generator, Optional, Any
import pandas as pd
import numpy as np

from src.data.data_lake import DataLake
from src.core.events import MarketEvent
from src.market_data.provider import MarketDataProvider


class ReplayMarketDataProvider(MarketDataProvider):
    """
    Historical market data provider that streams synchronized bar events from DataLake.
    Guarantees zero future-leakage by emitting bars one time-slice at a time.
    """

    def __init__(
        self,
        data_lake: Optional[DataLake] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.lake = data_lake or DataLake(read_only=True)
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        self.subscribed_symbols: List[str] = []
        self.timeframe: str = "15m"
        self._full_bars: Dict[str, pd.DataFrame] = {}
        self._stream_bars: Dict[str, pd.DataFrame] = {}
        self._latest_prices: Dict[str, float] = {}

    def subscribe(self, symbols: List[str], timeframe: str = "15m"):
        """Subscribes to historical data for symbols and timeframe."""
        self.timeframe = timeframe
        for sym in symbols:
            s_clean = sym.upper()
            if s_clean not in self.subscribed_symbols:
                self.subscribed_symbols.append(s_clean)
                df = self.lake.load_bars(s_clean, timeframe)
                if not df.empty:
                    if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.set_index("timestamp").sort_index()
                    
                    self._full_bars[s_clean] = df

                    stream_df = df
                    if self.start_date is not None:
                        stream_df = stream_df.loc[stream_df.index >= self.start_date]
                    if self.end_date is not None:
                        stream_df = stream_df.loc[stream_df.index <= self.end_date]
                    
                    self._stream_bars[s_clean] = stream_df

    def get_warmup_bars(self, symbol: str, count: int = 800) -> List[Dict[str, Any]]:
        """Returns the last `count` bars immediately prior to start_date for warmup."""
        s_clean = symbol.upper()
        df = self._full_bars.get(s_clean, pd.DataFrame())
        if df.empty or self.start_date is None:
            return []
        
        prior_df = df.loc[df.index < self.start_date]
        if prior_df.empty:
            return []
        
        warmup_df = prior_df.tail(count)
        records = []
        for ts, row in warmup_df.iterrows():
            vwap_val = float(row["vwap"]) if "vwap" in row and pd.notna(row["vwap"]) else None
            records.append({
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0)),
                "vwap": vwap_val,
            })
        return records

    def stream_events(self) -> Generator[MarketEvent, None, None]:
        """
        Merges and streams bars across all subscribed symbols in strict chronological order.
        """
        if not self._stream_bars:
            return

        # Collect all rows with (timestamp, symbol, row)
        records = []
        for sym, df in self._stream_bars.items():
            for ts, row in df.iterrows():
                records.append((ts, sym, row))

        # Strict chronological sort by (timestamp, symbol)
        records.sort(key=lambda x: (x[0], x[1]))

        for ts, sym, row in records:
            close_p = float(row["close"])
            self._latest_prices[sym] = close_p
            
            vwap_val = float(row["vwap"]) if "vwap" in row and pd.notna(row["vwap"]) else None

            yield MarketEvent(
                symbol=sym,
                timestamp=ts,
                timeframe=self.timeframe,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=close_p,
                volume=int(row.get("volume", 0)),
                vwap=vwap_val,
            )

    def get_latest_price(self, symbol: str) -> Optional[float]:
        return self._latest_prices.get(symbol.upper(), None)

    def get_historical_slice(self, symbol: str, up_to_timestamp: pd.Timestamp) -> pd.DataFrame:
        """Returns historical bars strictly up to the current simulation time (point-in-time window)."""
        sym_clean = symbol.upper()
        df = self._full_bars.get(sym_clean, pd.DataFrame())
        if df.empty:
            return df
        return df.loc[:up_to_timestamp]
