"""
Ashva Angel One SmartAPI Historical Data Client
Handles automated TOTP session generation, token lookup, and historical bar retrieval.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import pyotp
import requests
from src.data.data_lake import DataLake


class DataIntegrityError(RuntimeError):
    """Raised when historical market data cannot be retrieved or verified."""
    pass


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
        Downloads and caches the official Angel One Instrument Scrip Master to disk and memory.
        """
        if self._scrip_master is not None:
            return self._scrip_master

        cache_file = Path("data_lake/scrip_master.parquet")
        if cache_file.exists() and cache_file.stat().st_size > 100000:
            try:
                self._scrip_master = pd.read_parquet(cache_file)
                return self._scrip_master
            except Exception:
                pass

        import json
        for attempt in range(4):
            try:
                with requests.get(self.SCRIP_MASTER_URL, stream=True, timeout=90) as r:
                    r.raise_for_status()
                    data = r.json()
                    self._scrip_master = pd.DataFrame(data)
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    self._scrip_master.to_parquet(cache_file, index=False)
                    return self._scrip_master
            except Exception as e:
                import time as _t
                _t.sleep(2.0 * (attempt + 1))
                if attempt == 3:
                    if cache_file.exists():
                        self._scrip_master = pd.read_parquet(cache_file)
                        return self._scrip_master
                    raise ConnectionError(f"Could not load Scrip Master from Angel One: {e}")

        return self._scrip_master

    def get_token_for_symbol(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """
        Finds the Angel One instrument token for a given symbol.
        """
        scrips = self.load_scrip_master()
        sym_clean = symbol.upper()
        # Aliases for renamed/demerged corporate actions
        alias_map = {
            "TATAMOTORS": "TMPV",
            "M&M": "M&M",
            "BAJAJ-AUTO": "BAJAJ-AUTO",
            "ZOMATO": "ETERNAL",
        }
        target_sym = alias_map.get(sym_clean, sym_clean)

        match = scrips[(scrips["symbol"] == f"{target_sym}-EQ") & (scrips["exch_seg"] == exchange.upper())]
        if not match.empty:
            return str(match.iloc[0]["token"])
        
        # Fallback: search by name
        match_name = scrips[(scrips["name"] == target_sym) & (scrips["exch_seg"] == exchange.upper())]
        if not match_name.empty:
            return str(match_name.iloc[0]["token"])
        return None

    def fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        from_date: Any,  # String or datetime
        to_date: Any,    # String or datetime
        token: Optional[str] = None,
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        """
        Fetches historical candles from SmartAPI and stores in DataLake.
        """
        if self.smart_api is None:
            self.initialize_session()

        if not token:
            token = self.get_token_for_symbol(symbol, exchange)
            if not token:
                raise ValueError(f"Could not find Angel One instrument token for {symbol} on {exchange}")

        interval = self.INTERVAL_MAP.get(timeframe.lower())
        if not interval:
            raise ValueError(f"Unsupported timeframe {timeframe}. Supported: {list(self.INTERVAL_MAP.keys())}")

        # Format dates to "YYYY-MM-DD HH:MM"
        f_str = from_date.strftime("%Y-%m-%d %H:%M") if isinstance(from_date, datetime) else str(from_date)
        t_str = to_date.strftime("%Y-%m-%d %H:%M") if isinstance(to_date, datetime) else str(to_date)

        params = {
            "exchange": exchange.upper(),
            "symboltoken": token,
            "interval": interval,
            "fromdate": f_str,
            "todate": t_str,
        }

        # Angel One Rate Limit Protection: 3 requests/sec
        import time as _t
        _t.sleep(0.8)

        candle_response = None
        last_err = None
        for attempt in range(3):
            try:
                candle_response = self.smart_api.getCandleData(params)
                if isinstance(candle_response, dict) and candle_response.get("status"):
                    break
                else:
                    last_err = candle_response.get("message", "Unknown API error") if isinstance(candle_response, dict) else "Non-dict response"
            except Exception as e:
                last_err = str(e)
                _t.sleep(1.5 * (attempt + 1))

        if not isinstance(candle_response, dict) or not candle_response.get("status"):
            raise DataIntegrityError(
                f"CRITICAL: Failed to fetch {symbol} ({timeframe}) from {f_str} to {t_str} from Angel One SmartAPI. "
                f"Last error: {last_err}"
            )

        raw_data = candle_response.get("data", [])
        if not raw_data:
            raise DataIntegrityError(f"CRITICAL: Empty candle payload returned for {symbol} ({timeframe}) from {f_str} to {t_str}")

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
