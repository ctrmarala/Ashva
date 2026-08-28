"""
Ashva Data Lake Engine
Manages point-in-time market data storage and querying using DuckDB and Apache Parquet.
"""

import os
from pathlib import Path
from typing import Optional, List
import pandas as pd
import duckdb


class DataLake:
    """
    Columnar storage engine optimized for financial time series research and backtesting.
    """

    def __init__(self, db_path: str = "data_lake/ashva_market_data.duckdb", parquet_dir: str = "data_lake/parquet/", read_only: bool = False):
        self.db_path = Path(db_path)
        self.parquet_dir = Path(parquet_dir)
        self.read_only = read_only
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.conn = duckdb.connect(str(self.db_path), read_only=read_only)
        except Exception:
            try:
                self.conn = duckdb.connect(str(self.db_path), read_only=True)
            except Exception:
                self.conn = None

        if self.conn is not None and not read_only:
            try:
                self._initialize_schema()
            except Exception:
                pass

    def _initialize_schema(self):
        """Creates the primary OHLCV table if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_bars (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                timeframe VARCHAR NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT NOT NULL,
                source VARCHAR DEFAULT 'HISTORICAL',
                PRIMARY KEY (symbol, timestamp, timeframe)
            );
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup 
            ON ohlcv_bars (symbol, timeframe, timestamp);
        """)

    def save_bars(self, df: pd.DataFrame, symbol: str, timeframe: str, source: str = "HISTORICAL"):
        """
        Saves or updates a pandas DataFrame of OHLCV bars into DuckDB and Parquet.
        Expected columns: ['timestamp' (or index), 'open', 'high', 'low', 'close', 'volume']
        """
        if df.empty:
            return

        df_to_save = df.copy()
        
        # Normalize column names to lowercase
        df_to_save.columns = [str(col).lower() for col in df_to_save.columns]
        
        if "timestamp" not in df_to_save.columns:
            if isinstance(df_to_save.index, pd.DatetimeIndex):
                df_to_save["timestamp"] = df_to_save.index
            else:
                raise ValueError("DataFrame must contain a 'timestamp' column or DatetimeIndex")

        df_to_save["symbol"] = symbol.upper()
        df_to_save["timeframe"] = timeframe.lower()
        df_to_save["source"] = source
        df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"])

        required_cols = ["symbol", "timestamp", "timeframe", "open", "high", "low", "close", "volume", "source"]
        df_to_save = df_to_save[required_cols]

        # Upsert into DuckDB
        self.conn.register("temp_bars", df_to_save)
        self.conn.execute("""
            INSERT OR REPLACE INTO ohlcv_bars 
            SELECT * FROM temp_bars;
        """)
        self.conn.unregister("temp_bars")

        # Save partitioned Parquet backup (preserving historical continuity)
        parquet_file = self.parquet_dir / f"{symbol.upper()}_{timeframe.lower()}.parquet"
        if parquet_file.exists():
            try:
                existing_p = pd.read_parquet(parquet_file)
                combined = pd.concat([existing_p, df_to_save], ignore_index=True)
                combined.drop_duplicates(subset=["symbol", "timeframe", "timestamp"], keep="last", inplace=True)
                combined.sort_values(by="timestamp", inplace=True)
                combined.to_parquet(parquet_file, engine="pyarrow", index=False)
            except Exception:
                df_to_save.to_parquet(parquet_file, engine="pyarrow", index=False)
        else:
            df_to_save.to_parquet(parquet_file, engine="pyarrow", index=False)

    def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        max_lookback_days: int = 540,
    ) -> pd.DataFrame:
        """
        Loads OHLCV bars as a clean Pandas DataFrame with DatetimeIndex.
        Enforces Ashva's 540-day (18-Month) maximum research lookback ceiling.
        """
        df = pd.DataFrame()
        if self.conn is None:
            parquet_file = self.parquet_dir / f"{symbol.upper()}_{timeframe.lower()}.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    if start_time:
                        df = df[df["timestamp"] >= start_time]
                    if end_time:
                        df = df[df["timestamp"] <= end_time]
                    df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)
        else:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_bars
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol.upper(), timeframe.lower()]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp ASC"

            df = self.conn.execute(query, params).df()
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)

        if not df.empty and max_lookback_days > 0:
            max_ts = df.index.max()
            hard_cutoff = max_ts - pd.Timedelta(days=max_lookback_days)
            df = df[df.index >= hard_cutoff]

        return df

    def list_symbols(self, timeframe: Optional[str] = None) -> List[str]:
        """Returns list of unique symbols available in data lake."""
        if timeframe:
            res = self.conn.execute("SELECT DISTINCT symbol FROM ohlcv_bars WHERE timeframe = ?", [timeframe.lower()]).fetchall()
        else:
            res = self.conn.execute("SELECT DISTINCT symbol FROM ohlcv_bars").fetchall()
        return [row[0] for row in res]

    def close(self):
        """Closes DuckDB connection."""
        self.conn.close()
