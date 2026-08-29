import pytest
import pandas as pd
import sqlite3
from pathlib import Path
from src.ui.data_access import UIDataAccess

def test_alpha_registry_summary(tmp_path):
    exp_db = tmp_path / "exp.db"
    trd_db = tmp_path / "trd.db"
    
    # Create mock experiment ledger
    with sqlite3.connect(exp_db) as conn:
        conn.execute("""
            CREATE TABLE experiments (
                strategy_id TEXT,
                status TEXT,
                in_sample_sharpe REAL,
                cpcv_oos_sharpe REAL,
                deflated_sharpe_p_value REAL,
                net_profit_factor REAL,
                monte_carlo_95_max_dd REAL,
                trials_in_experiment INTEGER,
                timestamp TEXT
            )
        """)
        conn.execute("INSERT INTO experiments VALUES ('alpha_01', 'CAPITAL_CANDIDATE', 1.5, 1.2, 0.01, 1.8, -10.5, 100, '2023-01-01T00:00:00')")
        
    dal = UIDataAccess(exp_db_path=str(exp_db), trd_db_path=str(trd_db))
    df = dal.get_alpha_registry_summary()
    
    assert not df.empty
    assert "alpha_id" in df.columns
    # Check that dynamic status was joined correctly for alpha_01
    alpha_01_row = df[df["alpha_id"] == "alpha_01"]
    if not alpha_01_row.empty:
        assert alpha_01_row.iloc[0]["dynamic_status"] == "CAPITAL_CANDIDATE"

def test_missing_dbs(tmp_path):
    # Ensure it works when DBs are missing (returns static registry only)
    dal = UIDataAccess(exp_db_path=str(tmp_path / "missing_exp.db"), trd_db_path=str(tmp_path / "missing_trd.db"))
    
    df_alphas = dal.get_alpha_registry_summary()
    assert isinstance(df_alphas, pd.DataFrame)
    
    df_trading = dal.get_trading_state("REPLAY")
    assert df_trading.empty
    
    df_diag = dal.get_replay_diagnostics()
    assert df_diag.empty
