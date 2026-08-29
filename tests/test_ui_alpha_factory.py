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
def mock_dal(tmp_path):
    exp_db = tmp_path / "experiment_ledger.db"
    trd_db = tmp_path / "trading_ledger.db"
    duck_db = tmp_path / "ashva_market_data.duckdb"
    parquet_dir = tmp_path / "parquet"
    logs_dir = tmp_path / "logs"
    
    parquet_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(exp_db) as conn:
        conn.execute("""
            CREATE TABLE experiments (
                strategy_id TEXT,
                hypothesis_name TEXT,
                status TEXT,
                in_sample_sharpe REAL,
                cpcv_oos_sharpe REAL,
                deflated_sharpe_p_value REAL,
                net_profit_factor REAL,
                monte_carlo_95_max_dd REAL,
                category TEXT,
                timeframe TEXT,
                symbol_universe TEXT,
                trials_in_experiment INTEGER,
                timestamp TEXT,
                git_commit_sha TEXT
            )
        """)
        conn.execute("""
            INSERT INTO experiments VALUES 
            ('alpha_test_proven', 'Proven Test Alpha', 'PROVEN', 2.1, 1.6, 0.005, 1.85, -8.2, 'MOMENTUM', '15m', 'INFY,TCS,RELIANCE', 10, '2026-08-29T10:00:00', 'git123'),
            ('alpha_test_failed', 'Failed Test Alpha', 'FAILED', 0.4, -0.2, 0.45, 0.72, -22.5, 'MEAN_REVERSION', '15m', 'INFY,TCS', 5, '2026-08-29T11:00:00', 'git124'),
            ('alpha_test_untested', 'Untested Test Alpha', 'UNTESTED', 0.0, 0.0, 1.0, 0.0, 0.0, 'VOLATILITY_EXPANSION', '15m', 'INFY', 0, '2026-08-29T12:00:00', 'git125')
        """)

    return UIDataAccess(
        exp_db_path=str(exp_db),
        trd_db_path=str(trd_db),
        duckdb_path=str(duck_db),
        parquet_dir=str(parquet_dir),
        logs_dir=str(logs_dir),
    )


def test_alpha_factory_summary_counts(mock_dal):
    summary = mock_dal.get_alpha_factory_summary()
    assert summary["total_alphas"] >= 3
    assert summary["tested"] >= 2
    assert summary["proven"] == 1
    assert summary["failed"] >= 1
    assert summary["untested"] >= 1


def test_alpha_registry_table_structure(mock_dal):
    df = mock_dal.get_alpha_registry_table()
    assert not df.empty
    assert len(df) >= 3
    
    expected_cols = [
        "alpha_id", "name", "version", "status", "raw_status", "dynamic_status", "tested",
        "category", "economic_rationale", "timeframe", "universe", "test_period", "trades", "win_rate", "net_pnl",
        "expectancy", "profit_factor", "sharpe", "max_drawdown", "oos_trades", "oos_pnl",
        "oos_sharpe", "positive_symbols", "trials_count", "last_tested"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"


def test_proven_alpha_detail(mock_dal):
    detail = mock_dal.get_alpha_detail("alpha_test_proven")
    assert detail["alpha_id"] == "alpha_test_proven"
    assert detail["status"] == "PROVEN"
    assert detail["is_tested"] is True
    
    gates = detail["qualification_gates"]
    assert "gate_1_dsr" in gates
    assert "gate_2_cpcv" in gates
    assert "gate_3_mc_tail" in gates
    assert "gate_4_net_pf" in gates
    
    assert detail["metrics"]["profit_factor"] == 1.85
    assert "symbol_performance" in detail
    assert "research_evidence" in detail
    assert "replay_context" in detail
    assert "provenance" in detail


def test_failed_alpha_detail(mock_dal):
    detail = mock_dal.get_alpha_detail("alpha_test_failed")
    assert detail["alpha_id"] == "alpha_test_failed"
    assert detail["status"] == "FAILED"
    assert detail["is_tested"] is True


def test_untested_alpha_detail(mock_dal):
    detail = mock_dal.get_alpha_detail("alpha_test_untested")
    assert detail["alpha_id"] == "alpha_test_untested"
    assert detail["status"] == "UNTESTED"


def test_metrics_demarcation(mock_dal):
    detail = mock_dal.get_alpha_detail("alpha_test_proven")
    m = detail["metrics"]
    assert m["expectancy"] in ["NOT IMPLEMENTED", "NOT AVAILABLE"] or "Rs" in str(m["expectancy"])
    assert m["sortino"] == "NOT IMPLEMENTED"
    assert m["avg_win"] == "NOT IMPLEMENTED"


def test_promote_alpha_to_paper(mock_dal):
    # Proven alpha can be promoted
    success, msg = mock_dal.promote_alpha_to_paper("alpha_test_proven")
    assert success is True
    assert "Successfully promoted" in msg

    # Failed alpha cannot be promoted
    success_f, msg_f = mock_dal.promote_alpha_to_paper("alpha_test_failed")
    assert success_f is False
    assert "Cannot promote" in msg_f


def test_fresh_dal_behavior():
    dal1 = UIDataAccess()
    dal2 = UIDataAccess()
    assert dal1 is not dal2
    assert dal1.get_alpha_factory_summary() == dal2.get_alpha_factory_summary()


def test_missing_alpha_handling(mock_dal):
    detail = mock_dal.get_alpha_detail("non_existent_alpha_xyz")
    assert detail == {}
