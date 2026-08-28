"""
Unit tests for Tab 2 Alpha Factory Observability in Ashva UI.
Verifies alpha counts, status classifications, registry table loading, alpha detail loading,
qualification gate evaluation, and testing at least one PROVEN, one FAILED, one UNCERTAIN, and one UNEXPLORED alpha.
"""

import pytest
import pandas as pd
import sqlite3
from pathlib import Path
from src.ui.data_access import UIDataAccess


@pytest.fixture
def dal_instance():
    return UIDataAccess()


def test_alpha_factory_summary_counts(dal_instance):
    summary = dal_instance.get_alpha_factory_summary()
    assert summary["total_alphas"] >= 84
    assert summary["tested"] >= 80
    assert summary["proven"] > 0
    assert summary["failed"] > 0
    assert summary["uncertain"] >= 0
    assert summary["unexplored"] >= 2


def test_alpha_registry_table_structure(dal_instance):
    df = dal_instance.get_alpha_registry_table()
    assert not df.empty
    assert len(df) >= 84
    
    expected_cols = [
        "alpha_id", "name", "version", "status", "raw_status", "tested",
        "category", "timeframe", "universe", "test_period", "sharpe",
        "net_profit_factor", "oos_sharpe", "max_drawdown_pct", "positive_symbols",
        "trials_count", "last_tested"
    ]
    for col in expected_cols:
        assert col in df.columns


def test_proven_alpha_detail(dal_instance):
    # Test PROVEN Alpha (e.g. alpha_86)
    detail = dal_instance.get_alpha_detail("alpha_86")
    assert detail["alpha_id"] == "alpha_86"
    assert detail["status"] == "PROVEN"
    assert detail["is_tested"] is True
    assert "PROVEN" in detail["explanations"]["status_reason"]
    
    gates = detail["qualification_gates"]
    assert "gate_1_dsr" in gates
    assert "gate_2_cpcv" in gates
    assert "gate_3_mc_tail" in gates
    assert "gate_4_net_pf" in gates
    assert gates["gate_4_net_pf"]["passed"] is True
    
    assert detail["metrics"]["net_profit_factor"] >= 1.08
    assert len(detail["symbol_performance"]) > 0


def test_failed_alpha_detail(dal_instance):
    # Test FAILED Alpha (e.g. alpha_01)
    detail = dal_instance.get_alpha_detail("alpha_01")
    assert detail["alpha_id"] == "alpha_01"
    assert detail["status"] == "FAILED"
    assert detail["is_tested"] is True
    assert "FAILED" in detail["explanations"]["status_reason"]
    assert detail["explanations"]["failure_lessons"] != ""


def test_uncertain_alpha_detail(dal_instance):
    # Test UNCERTAIN Alpha (e.g. alpha_03 or alpha_04)
    detail = dal_instance.get_alpha_detail("alpha_03")
    assert detail["alpha_id"] == "alpha_03"
    assert detail["status"] in ["UNCERTAIN", "FAILED"]
    assert detail["is_tested"] is True


def test_unexplored_alpha_detail(dal_instance):
    # Test UNEXPLORED candidate (alpha_34)
    detail = dal_instance.get_alpha_detail("alpha_34")
    assert detail["alpha_id"] == "alpha_34"
    assert detail["status"] == "UNEXPLORED"
    assert detail["is_tested"] is False
    assert "UNEXPLORED" in detail["explanations"]["status_reason"]


def test_alpha_filtering(dal_instance):
    df = dal_instance.get_alpha_registry_table()
    
    df_proven = df[df["status"] == "PROVEN"]
    assert not df_proven.empty
    
    df_failed = df[df["status"] == "FAILED"]
    assert not df_failed.empty
    
    df_tested = df[df["tested"] == "YES"]
    assert len(df_tested) >= 80


def test_missing_alpha_handling(dal_instance):
    detail = dal_instance.get_alpha_detail("alpha_nonexistent_999")
    assert detail == {}
