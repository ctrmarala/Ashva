"""
Unit tests for Tab 2 Alpha Factory Observability in Ashva UI.
Verifies alpha counts, status classifications, registry table loading, alpha detail loading,
qualification gate evaluation, metric demarcations (NOT AVAILABLE vs NOT IMPLEMENTED),
research evidence, replay context, and fresh DAL behavior.
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
    assert summary["total_alphas"] >= 86
    assert summary["tested"] >= 84
    assert summary["proven"] > 0
    assert summary["failed"] > 0
    assert summary["uncertain"] >= 0
    assert summary["unexplored"] >= 2


def test_alpha_registry_table_structure(dal_instance):
    df = dal_instance.get_alpha_registry_table()
    assert not df.empty
    assert len(df) >= 84
    
    expected_cols = [
        "alpha_id", "name", "version", "status", "raw_status", "dynamic_status", "tested",
        "category", "timeframe", "universe", "test_period", "trades", "win_rate", "net_pnl",
        "expectancy", "profit_factor", "sharpe", "max_drawdown", "oos_trades", "oos_pnl",
        "oos_sharpe", "positive_symbols", "trials_count", "last_tested"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"


def test_proven_alpha_detail(dal_instance):
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
    
    assert detail["metrics"]["profit_factor"] >= 1.08
    assert len(detail["symbol_performance"]) > 0
    assert "research_evidence" in detail
    assert "replay_context" in detail
    assert "provenance" in detail


def test_failed_alpha_detail(dal_instance):
    detail = dal_instance.get_alpha_detail("alpha_01")
    assert detail["alpha_id"] == "alpha_01"
    assert detail["status"] == "FAILED"
    assert detail["is_tested"] is True
    assert "FAILED" in detail["explanations"]["status_reason"]
    assert detail["explanations"]["failure_lessons"] != ""


def test_uncertain_alpha_detail(dal_instance):
    detail = dal_instance.get_alpha_detail("alpha_03")
    assert detail["alpha_id"] == "alpha_03"
    assert detail["status"] in ["UNCERTAIN", "FAILED"]
    assert detail["is_tested"] is True


def test_unexplored_alpha_detail(dal_instance):
    detail = dal_instance.get_alpha_detail("alpha_34")
    assert detail["alpha_id"] == "alpha_34"
    assert detail["status"] == "UNEXPLORED"
    assert detail["is_tested"] is False
    assert "UNEXPLORED" in detail["explanations"]["status_reason"]


def test_metrics_demarcation(dal_instance):
    detail = dal_instance.get_alpha_detail("alpha_86")
    m = detail["metrics"]
    # Verify that unimplemented metrics are explicitly marked NOT IMPLEMENTED
    assert m["expectancy"] == "NOT IMPLEMENTED"
    assert m["sortino"] == "NOT IMPLEMENTED"
    assert m["avg_win"] == "NOT IMPLEMENTED"


def test_fresh_dal_behavior():
    dal1 = UIDataAccess()
    dal2 = UIDataAccess()
    assert dal1 is not dal2
    assert dal1.get_alpha_factory_summary() == dal2.get_alpha_factory_summary()


def test_missing_alpha_handling(dal_instance):
    detail = dal_instance.get_alpha_detail("alpha_nonexistent_999")
    assert detail == {}
