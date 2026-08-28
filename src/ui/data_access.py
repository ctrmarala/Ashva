import sqlite3
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

from src.research.knowledge_map import AlphaKnowledgeMap

class UIDataAccess:
    def __init__(self, exp_db_path="data_lake/experiment_ledger.db", trd_db_path="data_lake/trading_ledger.db"):
        self.exp_db_path = Path(exp_db_path)
        self.trd_db_path = Path(trd_db_path)
        self.knowledge_map = AlphaKnowledgeMap()

    def get_alpha_registry_summary(self) -> pd.DataFrame:
        """
        Merges static AlphaKnowledgeMap baseline with the latest dynamic results from experiment_ledger.db.
        """
        # 1. Get baseline knowledge map
        km_alphas = self.knowledge_map.get_all_mechanisms()
        df_km = pd.DataFrame([a.__dict__ for a in km_alphas])

        # 2. Query SQLite experiment ledger for dynamic test results
        if not self.exp_db_path.exists():
            return df_km

        try:
            with sqlite3.connect(self.exp_db_path) as conn:
                query = """
                    SELECT strategy_id, status as dynamic_status, in_sample_sharpe, 
                           cpcv_oos_sharpe, deflated_sharpe_p_value, net_profit_factor, 
                           monte_carlo_95_max_dd, trials_in_experiment, timestamp
                    FROM experiments 
                    WHERE (strategy_id, timestamp) IN (
                        SELECT strategy_id, MAX(timestamp) 
                        FROM experiments 
                        GROUP BY strategy_id
                    )
                """
                df_exp = pd.read_sql_query(query, conn)
                
            if not df_exp.empty:
                # Merge dynamic on top of static
                df_merged = pd.merge(df_km, df_exp, left_on="alpha_id", right_on="strategy_id", how="left")
            else:
                df_merged = df_km
            return df_merged
        except Exception as e:
            print(f"Error reading experiment ledger: {e}")
            return df_km

    def get_trading_state(self, mode: str) -> pd.DataFrame:
        """
        Queries trading_ledger.db for open positions or historical trades by mode (REPLAY, PAPER, LIVE)
        """
        if not self.trd_db_path.exists():
            return pd.DataFrame()

        try:
            with sqlite3.connect(self.trd_db_path) as conn:
                query = "SELECT * FROM trades WHERE mode = ?"
                df = pd.read_sql_query(query, conn, params=(mode,))
                return df
        except Exception as e:
            print(f"Error reading trading ledger: {e}")
            return pd.DataFrame()

    def get_replay_diagnostics(self) -> pd.DataFrame:
        """
        Queries replay_diagnostics table for signal drop-offs during REPLAY mode.
        """
        if not self.trd_db_path.exists():
            return pd.DataFrame()
            
        try:
            with sqlite3.connect(self.trd_db_path) as conn:
                # Check if table exists
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='replay_diagnostics'")
                if cursor.fetchone():
                    df = pd.read_sql_query("SELECT * FROM replay_diagnostics ORDER BY timestamp DESC LIMIT 100", conn)
                    return df
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading replay diagnostics: {e}")
            return pd.DataFrame()
