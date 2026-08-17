"""
Ashva Yahoo Finance Historical Data Loader
Downloads historical market data for Indian equities and indices (NSE) with automated symbol formatting.
"""

from typing import Optional
import pandas as pd
import yfinance as yf
from src.data.data_lake import DataLake


class YFinanceLoader:
    """
    Utility to fetch and store historical OHLCV data for offline research and backtesting.
    """

    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "1h": "1h",
        "1d": "1d",
        "1wk": "1wk",
    }

    def __init__(self, data_lake: Optional[DataLake] = None):
        self.data_lake = data_lake or DataLake()

    def _format_ticker(self, symbol: str) -> str:
        """
        Converts Indian ticker symbols to Yahoo Finance format.
        e.g., RELIANCE -> RELIANCE.NS, NIFTY 50 -> ^NSEI
        """
        sym = symbol.strip().upper()
        if sym in ["NIFTY", "NIFTY50", "NIFTY 50", "^NSEI"]:
            return "^NSEI"
        if sym in ["BANKNIFTY", "^NSEBANK"]:
            return "^NSEBANK"
        if not sym.endswith(".NS") and not sym.endswith(".BO") and not sym.startswith("^"):
            return f"{sym}.NS"
        return sym

    def fetch_and_store(
        self,
        symbol: str,
        timeframe: str = "5m",
        period: str = "1mo",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Downloads data from Yahoo Finance and automatically saves it to the DataLake.
        
        :param symbol: Equity or Index symbol (e.g., 'RELIANCE', 'TCS', 'NIFTY')
        :param timeframe: '1m', '5m', '15m', '1h', '1d'
        :param period: Valid yfinance period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        :param start_date: 'YYYY-MM-DD'
        :param end_date: 'YYYY-MM-DD'
        """
        yf_ticker = self._format_ticker(symbol)
        tf = self.TIMEFRAME_MAP.get(timeframe.lower(), "5m")

        ticker_obj = yf.Ticker(yf_ticker)
        
        if start_date:
            df = ticker_obj.history(start=start_date, end=end_date, interval=tf)
        else:
            df = ticker_obj.history(period=period, interval=tf)

        if df.empty:
            raise ValueError(f"No data returned for {symbol} ({yf_ticker}) with timeframe {timeframe}")

        # Clean columns
        df.reset_index(inplace=True)
        # Rename Datetime / Date column to timestamp
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df.rename(columns={
            date_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        clean_df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()

        # Save to local Data Lake
        self.data_lake.save_bars(
            df=clean_df,
            symbol=symbol.upper(),
            timeframe=timeframe.lower(),
            source="YFINANCE"
        )

        return self.data_lake.load_bars(symbol.upper(), timeframe.lower())
