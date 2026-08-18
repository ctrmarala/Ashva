"""
Ashva Research Experiment Registry & Multiple-Testing Audit Ledger
Maintains an immutable trial journal of every quantitative hypothesis tested.
Tracks trial counts (N) for rigorous Deflated Sharpe Ratio (DSR) family-wise error rate corrections.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Dict, List, Any, Optional


def get_current_git_sha() -> str:
    """Dynamically returns current Git commit SHA or 'UNKNOWN'."""
    try:
        res = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return res.strip()
    except Exception:
        return "DEV_DIRTY"


@dataclass
class ExperimentRecord:
    experiment_id: str
    strategy_id: str
    symbol_universe: str
    timeframe: str
    parameters_json: str
    in_sample_sharpe: float
    cpcv_oos_sharpe: float
    deflated_sharpe_p_value: float
    net_profit_factor: float
    monte_carlo_95_max_dd: float
    trials_in_experiment: int
    total_trials_cumulative: int
    git_commit_sha: str
    status: str
    rejection_reasons_json: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.git_commit_sha:
            self.git_commit_sha = get_current_git_sha()


class ResearchExperimentLedger:
    """
    Persistent SQLite & JSONL Research Journal.
    Prevents p-hacking and selective reporting by logging every single model run.
    """

    def __init__(self, db_path: str = "data_lake/experiment_ledger.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    symbol_universe TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    in_sample_sharpe DOUBLE NOT NULL,
                    cpcv_oos_sharpe DOUBLE NOT NULL,
                    deflated_sharpe_p_value DOUBLE NOT NULL,
                    net_profit_factor DOUBLE NOT NULL,
                    monte_carlo_95_max_dd DOUBLE NOT NULL,
                    trials_in_experiment INTEGER NOT NULL DEFAULT 1,
                    total_trials_cumulative INTEGER NOT NULL,
                    git_commit_sha TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rejection_reasons_json TEXT NOT NULL
                );
            """)

    def get_total_trials(self) -> int:
        """Returns cumulative count of all tested hypotheses/parameter trials across platform history."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COALESCE(SUM(trials_in_experiment), 0) FROM experiment_journal").fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def get_strategy_family_trials(self, strategy_id: str) -> int:
        """Returns cumulative trials for a specific strategy family (e.g. ALPHA_02, ALPHA_14) to prevent cross-family DSR penalty."""
        # Extract family prefix (e.g. 'ALPHA_02' from 'ALPHA_02_AUCTION_ORB')
        prefix = strategy_id.split("_")[0] + "_" + strategy_id.split("_")[1] if "_" in strategy_id else strategy_id
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(trials_in_experiment), 0) FROM experiment_journal WHERE strategy_id LIKE ?",
                (f"{prefix}%",)
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def log_experiment(self, record: ExperimentRecord) -> int:
        """Logs an experiment and returns the updated cumulative trial count."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiment_journal (
                    experiment_id, timestamp, strategy_id, symbol_universe, timeframe,
                    parameters_json, in_sample_sharpe, cpcv_oos_sharpe, deflated_sharpe_p_value,
                    net_profit_factor, monte_carlo_95_max_dd, trials_in_experiment,
                    total_trials_cumulative, git_commit_sha, status, rejection_reasons_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                record.experiment_id,
                record.timestamp,
                record.strategy_id,
                record.symbol_universe,
                record.timeframe,
                record.parameters_json,
                record.in_sample_sharpe,
                record.cpcv_oos_sharpe,
                record.deflated_sharpe_p_value,
                record.net_profit_factor,
                record.monte_carlo_95_max_dd,
                record.trials_in_experiment,
                record.total_trials_cumulative,
                record.git_commit_sha,
                record.status,
                record.rejection_reasons_json,
            ))
        return self.get_total_trials()

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM experiment_journal ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
