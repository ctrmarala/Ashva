"""
Ashva Universe Management Engine
Provides universe-agnostic resolution of tradable assets, index benchmarks,
and broker instrument tokens (supporting NIFTY 50, NIFTY 100, NIFTY 200, NIFTY 500, or Custom).
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml
import duckdb


DEFAULT_NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN", "ITC", "HINDUNILVR",
    "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", "ADANIENT", "KOTAKBANK", "TATAMOTORS", "TATASTEEL",
    "NTPC", "AXISBANK", "ONGC", "TITAN", "ADANIPORTS", "COALINDIA", "POWERGRID", "M&M", "BAJAJFINSV", "WIPRO",
    "NESTLEIND", "ULTRACEMCO", "JSWSTEEL", "GRASIM", "TECHM", "EICHERMOT", "HDFCLIFE", "BPCL", "DRREDDY",
    "BRITANNIA", "CIPLA", "APOLLOHOSP", "TATACONSUM", "SHRIRAMFIN", "BAJAJ-AUTO", "BEL", "HEROMOTOCO",
    "INDUSINDBK", "SBILIFE", "TRENT", "DIVISLAB", "PIDILITIND"
]


class UniverseManager:
    """
    Centralized Universe Provider ensuring full decoupling from static symbol lists.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)

    def load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def get_universe_name(self) -> str:
        """Returns active universe name (e.g., 'NIFTY 50', 'NIFTY 100', 'NIFTY 500')."""
        cfg = self.load_config().get("universe", {})
        return cfg.get("name", "NIFTY 50")

    def get_benchmark_symbol(self) -> str:
        """Returns benchmark index ticker (e.g., '^NSEI')."""
        cfg = self.load_config().get("universe", {})
        return cfg.get("benchmark", "^NSEI")

    def get_universe_symbols(self, duckdb_path: Optional[str] = "data_lake/ashva_market_data.duckdb") -> List[str]:
        """
        Dynamically resolves the active tradable universe symbols in priority order:
        1. Explicit list in config/settings.yaml under universe.symbols
        2. Configured symbols JSON file (e.g. config/nifty50_tokens.json or config/universe_tokens.json)
        3. Stored symbols currently inside DataLake (DuckDB)
        4. Fallback default universe
        """
        cfg = self.load_config().get("universe", {})

        # 1. Explicit list in settings.yaml
        if "symbols" in cfg and isinstance(cfg["symbols"], list) and cfg["symbols"]:
            return sorted([s.upper() for s in cfg["symbols"]])

        # 2. Token mapping file
        token_file = cfg.get("token_file", "config/nifty50_tokens.json")
        token_path = Path(token_file)
        if token_path.exists():
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and data:
                        return sorted([k.upper() for k in data.keys()])
                    elif isinstance(data, list) and data:
                        return sorted([s.upper() for s in data])
            except Exception:
                pass

        # 3. Query DataLake DuckDB if available
        if duckdb_path and Path(duckdb_path).exists():
            try:
                conn = duckdb.connect(str(duckdb_path), read_only=True)
                rows = conn.execute("SELECT DISTINCT symbol FROM ohlcv_bars ORDER BY symbol ASC").fetchall()
                conn.close()
                if rows:
                    return [r[0].upper() for r in rows]
            except Exception:
                pass

        # 4. Fallback Default
        return sorted(DEFAULT_NIFTY_50)


# Global helper functions for drop-in usage across all scripts and UI modules
def get_universe_symbols(config_path: str = "config/settings.yaml", duckdb_path: Optional[str] = "data_lake/ashva_market_data.duckdb") -> List[str]:
    return UniverseManager(config_path=config_path).get_universe_symbols(duckdb_path=duckdb_path)

def get_universe_name(config_path: str = "config/settings.yaml") -> str:
    return UniverseManager(config_path=config_path).get_universe_name()

def get_benchmark_symbol(config_path: str = "config/settings.yaml") -> str:
    return UniverseManager(config_path=config_path).get_benchmark_symbol()
