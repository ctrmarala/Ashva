"""
Unit tests for NSE Holiday Calendar integration and One-Click Data Sync & Validation in Ashva UI.
"""

from datetime import date
import pytest
from src.data.nse_calendar import NSECalendar, ALL_NSE_HOLIDAYS
from src.ui.data_access import UIDataAccess


@pytest.fixture
def dal_instance():
    return UIDataAccess()


def test_nse_holiday_calendar_rules():
    # 1. Test weekend check
    assert not NSECalendar.is_trading_day(date(2026, 8, 15))  # Saturday
    assert not NSECalendar.is_trading_day(date(2026, 8, 16))  # Sunday

    # 2. Test official holidays
    assert not NSECalendar.is_trading_day(date(2026, 1, 26))  # Republic Day 2026
    assert not NSECalendar.is_trading_day(date(2025, 12, 25)) # Christmas 2025
    assert not NSECalendar.is_trading_day(date(2024, 10, 2))  # Gandhi Jayanti 2024

    # 3. Test regular trading day
    assert NSECalendar.is_trading_day(date(2026, 8, 28))      # Friday (Trading session)
    assert NSECalendar.is_trading_day(date(2026, 8, 27))      # Thursday (Trading session)


def test_nse_trading_days_range():
    # Between 2026-08-24 (Mon) and 2026-08-28 (Fri) = 5 trading days
    days = NSECalendar.get_trading_days("2026-08-24", "2026-08-28")
    assert len(days) == 5
    assert date(2026, 8, 24) in days
    assert date(2026, 8, 28) in days


def test_expected_bars_count():
    # 5 trading days * 25 bars/day (15m) = 125 bars
    expected_bars = NSECalendar.get_expected_bars_count("2026-08-24", "2026-08-28", timeframe="15m")
    assert expected_bars == 125

    # 5 trading days * 1 bar/day (1d) = 5 bars
    expected_daily_bars = NSECalendar.get_expected_bars_count("2026-08-24", "2026-08-28", timeframe="1d")
    assert expected_daily_bars == 5


def test_audit_symbol_calendar_coverage():
    audit = NSECalendar.audit_symbol_calendar_coverage("INFY", timeframe="15m")
    assert audit["symbol"] == "INFY"
    assert audit["timeframe"] == "15m"
    assert audit["expected_trading_days"] > 350
    assert audit["actual_trading_days"] > 350
    assert audit["coverage_pct"] >= 95.0
    assert "summary_text" in audit


def test_dal_symbol_detail_includes_calendar_audit(dal_instance):
    detail = dal_instance.get_symbol_detail("INFY")
    assert "calendar_audit" in detail
    assert "unadjusted_stock_splits" in detail["quality_metrics"]
    cal = detail["calendar_audit"]
    assert "expected_trading_days" in cal
    assert "coverage_pct" in cal
    assert "missing_trading_days_count" in cal


def test_dal_comprehensive_validation(dal_instance):
    val_report = dal_instance.run_comprehensive_data_validation()
    assert "hygiene_summary" in val_report
    assert "symbols_audited_count" in val_report
    assert val_report["symbols_audited_count"] > 0
    assert "calendar_audits_table" in val_report
    assert len(val_report["calendar_audits_table"]) > 0
    assert "split_audit_status" in val_report
    assert "split_anomalies_count" in val_report
    assert val_report["split_anomalies_count"] == 0


def test_dal_sync_market_data_now(dal_instance):
    # Test sync for sample symbol INFY (15m)
    res = dal_instance.sync_market_data_now(symbol="INFY", timeframe="15m", period="5d")
    assert res["status"] in ["SUCCESS", "WARNING", "IDLE"]
    assert res["symbols_requested"] == 1
    assert "provider_used" in res
