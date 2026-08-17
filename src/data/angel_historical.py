"""
Ashva Angel One SmartAPI Historical Data Client
Handles automated TOTP session generation, token lookup, and historical bar retrieval.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import pyotp
import requests
from src.data.data_lake import DataLake


class AngelHistoricalFetcher:
    """
    Client to fetch historical candles directly from Angel One SmartAPI and cache in DataLake.
    """

    SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

    INTERVAL_MAP = {
        "1m": "ONE_MINUTE",
        "3m": "THREE_MINUTE",
        "5m": "FIVE_MINUTE",
        "10m": "TEN_MINUTE",
        "15m": "FIFTEEN_MINUTE",
        "30m": "THIRTY_MINUTE",
        "60m": "ONE_HOUR",
        "1h": "ONE_HOUR",
        "1d": "ONE_DAY",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        client_code: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None,
        data_lake: Optional[DataLake] = None,
    ):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.data_lake = data_lake or DataLake()
        self.smart_api = None
        self._scrip_master: Optional[pd.DataFrame] = None
        self.auth_token: Optional[str] = None
        self.feed_token: Optional[str] = None

    def initialize_session(self):
        """
        Logs in to Angel One SmartAPI using TOTP.
        """
        if not all([self.api_key, self.client_code, self.password, self.totp_secret]):
            raise ValueError("Angel One API credentials (API Key, Client Code, Password, TOTP Secret) are required.")

        try:
            from SmartApi import SmartConnect
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            data = self.smart_api.generateSession(self.client_code, self.password, totp)
            
            if data.get("status"):
                self.auth_token = data["data"]["jwtToken"]
                self.feed_token = data["data"]["feedToken"]
                return data["data"]
            else:
                raise ConnectionError(f"SmartAPI login failed: {data.get('message')}")
        except ImportError:
            raise ImportError("smartapi-python library is required. Install via pip install smartapi-python")

    def load_scrip_master(self) -> pd.DataFrame:
        """
        Downloads and caches the official Angel One Instrument Scrip Master.
        """
        if self._scrip_master is None:
            response = requests.get(self.SCRIP_MASTER_URL, timeout=30)
            if response.status_code == 200:
                self._scrip_master = pd.DataFrame(response.json())
            else:
                raise ConnectionError(f"Failed to fetch Scrip Master: HTTP {response.status_code}")
        return self._scrip_master

    def get_token_for_symbol(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """
        Finds the Angel One instrument token for a given symbol.
        """
        scrips = self.load_scrip_master()
        match = scrips[(scrips["symbol"] == f"{symbol.upper()}-EQ") & (scrips["exch_seg"] == exchange.upper())]
        if not match.empty:
            return str(match.iloc[0]["token"])
        
        # Fallback: search by name
        match_name = scrips[(scrips["name"] == symbol.upper()) & (scrips["exch_seg"] == exchange.upper())]
        if not match_name.empty:
            return str(match_name.iloc[0]["token"])
        return None

    def fetch_and_store(
        self,
        symbol: str,
        token: str,
        timeframe: str,
        from_date: str,  # Format: "YYYY-MM-DD HH:MM"
        to_date: str,    # Format: "YYYY-MM-DD HH:MM"
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        """
        Fetches historical candles from SmartAPI and stores in DataLake.
        """
        if self.smart_api is None:
            self.initialize_session()

        interval = self.INTERVAL_MAP.get(timeframe.lower())
        if not interval:
            raise ValueError(f"Unsupported timeframe {timeframe}. Supported: {list(self.INTERVAL_MAP.keys())}")

        params = {
            "exchange": exchange.upper(),
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }

        # Angel One Rate Limit Protection: 3 requests/sec
        import time as _t
        _t.sleep(0.8)

        candle_response = None
        for attempt in range(3):
            try:
                candle_response = self.smart_api.getCandleData(params)
                if isinstance(candle_response, dict) and candle_response.get("status"):
                    break
            except Exception:
                _t.sleep(1.5 * (attempt + 1))

        if not isinstance(candle_response, dict) or not candle_response.get("status"):
            return self.data_lake.load_bars(symbol.upper(), timeframe.lower())

        raw_data = candle_response.get("data", [])
        if not raw_data:
            return pd.DataFrame()

        df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(int)

        # Store in DataLake
        self.data_lake.save_bars(
            df=df,
            symbol=symbol.upper(),
            timeframe=timeframe.lower(),
            source="ANGEL_ONE"
        )

        return self.data_lake.load_bars(symbol.upper(), timeframe.lower())
