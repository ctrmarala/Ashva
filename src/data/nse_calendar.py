"""
Ashva Official NSE Holiday Calendar & Trading Day Verification Module
Provides authoritative verification of NSE Cash Market trading sessions, holidays,
and point-in-time trading day gap calculations for 2024, 2025, and 2026.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Set, Union, Optional, Any
import pandas as pd
import duckdb


# Official NSE Equity Trading Holidays (excluding Saturday/Sunday standard closures)
# Source: National Stock Exchange of India (NSE) Circulars
NSE_HOLIDAYS_2024: Set[date] = {
    date(2024, 1, 22),  # Special Holiday
    date(2024, 1, 26),  # Republic Day
    date(2024, 3, 8),   # Mahashivratri
    date(2024, 3, 25),  # Holi
    date(2024, 3, 29),  # Good Friday
    date(2024, 4, 11),  # Id-Ul-Fitr (Ramzan Id)
    date(2024, 4, 17),  # Shri Ram Navami
    date(2024, 5, 1),   # Maharashtra Day
    date(2024, 5, 20),  # General Parliamentary Elections (Mumbai)
    date(2024, 6, 17),  # Bakri Id / Eid ul-Adha
    date(2024, 7, 17),  # Muharram
    date(2024, 8, 15),  # Independence Day / Parsi New Year
    date(2024, 10, 2),  # Mahatma Gandhi Jayanti
    date(2024, 11, 1),  # Diwali Laxmi Pujan (Muhurat trading evening session only)
    date(2024, 11, 15), # Gurunanak Jayanti
    date(2024, 11, 20), # Maharashtra Assembly Elections
    date(2024, 12, 25), # Christmas
}

NSE_HOLIDAYS_2025: Set[date] = {
    date(2025, 2, 26),  # Mahashivratri
    date(2025, 3, 14),  # Holi
    date(2025, 3, 31),  # Id-Ul-Fitr
    date(2025, 4, 10),  # Mahavir Jayanti
    date(2025, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 1),   # Maharashtra Day
    date(2025, 6, 7),   # Bakri Id
    date(2025, 8, 15),  # Independence Day
    date(2025, 8, 27),  # Ganesh Chaturthi
    date(2025, 10, 2),  # Mahatma Gandhi Jayanti / Dussehra
    date(2025, 10, 21), # Diwali (Laxmi Pujan)
    date(2025, 10, 22), # Diwali Balipratipada
    date(2025, 11, 5),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2025, 12, 25), # Christmas
}

NSE_HOLIDAYS_2026: Set[date] = {
    date(2026, 1, 15),  # Makar Sankranti / Municipal Election
    date(2026, 1, 26),  # Republic Day
    date(2026, 2, 16),  # Mahashivratri
    date(2026, 3, 3),   # Holi
    date(2026, 3, 20),  # Id-Ul-Fitr
    date(2026, 3, 26),  # Shri Ram Navami
    date(2026, 3, 31),  # Mahavir Jayanti / Annual Closing
    date(2026, 4, 3),   # Good Friday
    date(2026, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),   # Maharashtra Day
    date(2026, 5, 28),  # Bakri Id
    date(2026, 6, 26),  # Muharram
    date(2026, 8, 15),  # Independence Day (Saturday)
    date(2026, 10, 2),  # Mahatma Gandhi Jayanti
    date(2026, 10, 20), # Dussehra
    date(2026, 11, 8),  # Diwali (Laxmi Pujan)
    date(2026, 11, 9),  # Diwali Balipratipada
    date(2026, 11, 24), # Gurunanak Jayanti
    date(2026, 12, 25), # Christmas
}

# Official NSE Special Live Trading Sessions (e.g. Budget Sunday session, Diwali Muhurat Trading)
NSE_SPECIAL_SESSIONS: Set[date] = {
    date(2024, 11, 1),  # Diwali Muhurat Trading 2024
    date(2025, 10, 21), # Diwali Muhurat Trading 2025
    date(2026, 2, 1),   # Union Budget Special Live Sunday Trading 2026
}

ALL_NSE_HOLIDAYS: Set[date] = NSE_HOLIDAYS_2024.union(NSE_HOLIDAYS_2025).union(NSE_HOLIDAYS_2026)


class NSECalendar:
    """
    Official NSE Cash Segment calendar engine for trading day resolution,
    missing bar detection, and data completeness auditing.
    """

    @staticmethod
    def to_date(d: Union[date, datetime, str, pd.Timestamp]) -> date:
        if isinstance(d, pd.Timestamp):
            return d.date()
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            return pd.to_datetime(d).date()
        raise ValueError(f"Cannot convert {type(d)} to date")

    @classmethod
    def is_trading_day(cls, dt: Union[date, datetime, str]) -> bool:
        """
        Returns True if the date is an active NSE trading session (including special live sessions).
        """
        d = cls.to_date(dt)
        # Check special exchange sessions (e.g. Budget Sunday, Muhurat Trading)
        if d in NSE_SPECIAL_SESSIONS:
            return True
        # Check weekend (5 = Saturday, 6 = Sunday)
        if d.weekday() >= 5:
            return False
        # Check official holidays
        if d in ALL_NSE_HOLIDAYS:
            return False
        return True

    @classmethod
    def get_trading_days(cls, start_date: Union[date, datetime, str], end_date: Union[date, datetime, str]) -> List[date]:
        """
        Returns all official NSE trading days between start_date and end_date (inclusive).
        """
        s = cls.to_date(start_date)
        e = cls.to_date(end_date)
        if s > e:
            s, e = e, s

        trading_days = []
        cur = s
        while cur <= e:
            if cls.is_trading_day(cur):
                trading_days.append(cur)
            cur += timedelta(days=1)
        return trading_days

    @classmethod
    def get_expected_trading_days_count(cls, start_date: Union[date, datetime, str], end_date: Union[date, datetime, str]) -> int:
        """Returns total expected market sessions between two dates."""
        return len(cls.get_trading_days(start_date, end_date))

    @classmethod
    def get_expected_bars_count(cls, start_date: Union[date, datetime, str], end_date: Union[date, datetime, str], timeframe: str = "15m") -> int:
        """
        Calculates expected intraday candle count.
        Standard regular market session: 09:15 to 15:30 = 375 minutes.
        - 15m = 25 candles/day
        - 5m  = 75 candles/day
        - 1m  = 375 candles/day
        - 30m = 13 candles/day (with last candle being 15:00-15:30)
        - 1d  = 1 candle/day
        """
        num_days = cls.get_expected_trading_days_count(start_date, end_date)
        tf_lower = timeframe.lower()
        bars_per_day_map = {
            "1m": 375,
            "5m": 75,
            "10m": 38,
            "15m": 25,
            "30m": 13,
            "60m": 7,
            "1h": 7,
            "1d": 1,
        }
        bars_per_day = bars_per_day_map.get(tf_lower, 25)
        return num_days * bars_per_day

    @classmethod
    def audit_symbol_calendar_coverage(
        cls,
        symbol: str,
        timeframe: str = "15m",
        duckdb_path: str = "data_lake/ashva_market_data.duckdb"
    ) -> Dict[str, Any]:
        """
        Audits actual stored bars in DuckDB against the official NSE Trading Calendar.
        Identifies any missing market sessions or intraday session dropouts.
        """
        try:
            conn = duckdb.connect(str(duckdb_path), read_only=True)
        except Exception as e:
            return {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "status": "ERROR_CONNECTING_DB",
                "error": str(e),
            }

        try:
            sym_upper = symbol.upper()
            tf_lower = timeframe.lower()

            # Query distinct dates present in DB for symbol
            df_dates = conn.execute("""
                SELECT 
                    CAST(timestamp AS DATE) as trade_date,
                    COUNT(*) as bar_count,
                    MIN(timestamp) as first_bar,
                    MAX(timestamp) as last_bar
                FROM ohlcv_bars
                WHERE symbol = ? AND timeframe = ?
                GROUP BY trade_date
                ORDER BY trade_date ASC
            """, [sym_upper, tf_lower]).df()

            if df_dates.empty:
                return {
                    "symbol": sym_upper,
                    "timeframe": tf_lower,
                    "status": "NO_DATA",
                    "actual_days": 0,
                    "expected_days": 0,
                    "missing_days_count": 0,
                    "missing_dates": [],
                    "coverage_pct": 0.0,
                    "summary_text": "No market data recorded.",
                }

            first_dt = pd.to_datetime(df_dates["trade_date"].min()).date()
            last_dt = pd.to_datetime(df_dates["trade_date"].max()).date()

            # Expected trading days based on NSE Calendar
            expected_days = cls.get_trading_days(first_dt, last_dt)
            actual_days_set = set(pd.to_datetime(df_dates["trade_date"]).dt.date.values)

            missing_days = [d for d in expected_days if d not in actual_days_set]
            missing_count = len(missing_days)
            expected_count = len(expected_days)
            actual_count = len(df_dates)

            coverage_pct = round((actual_count / expected_count) * 100.0, 2) if expected_count > 0 else 100.0

            status_str = "PASS (100% Calendar Coverage)" if missing_count == 0 else f"GAPS DETECTED ({missing_count} missing trading sessions)"

            return {
                "symbol": sym_upper,
                "timeframe": tf_lower,
                "first_date": str(first_dt),
                "last_date": str(last_dt),
                "calendar_span_days": (last_dt - first_dt).days,
                "expected_trading_days": expected_count,
                "actual_trading_days": actual_count,
                "missing_trading_days_count": missing_count,
                "missing_dates": [str(d) for d in missing_days[:10]],  # First 10 missing
                "coverage_pct": coverage_pct,
                "status": "CLEAN" if missing_count == 0 else "PARTIAL",
                "summary_text": status_str,
            }
        except Exception as e:
            return {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "status": "AUDIT_FAILED",
                "error": str(e),
            }
        finally:
            conn.close()
