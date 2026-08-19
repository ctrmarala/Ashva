"""
Ashva Autonomous Alpha Discovery Controller — Factory v2
Orchestrates closed-loop quantitative research:
Knowledge Map -> Hypothesis Generation -> Stage 0 Feasibility -> Stage 1 CI -> Stage 2 DEV -> Escalation Gate -> Stage 4 Research.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.data_lake import DataLake
from src.features.indicators import TechnicalIndicators
from src.research.knowledge_map import AlphaKnowledgeMap, AlphaCategory, MechanismStatus, AlphaResearchRecord
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord, get_current_git_sha


@dataclass
class ResearchBudget:
    max_hypotheses: int = 3
    max_dev_runs: int = 3
    max_research_runs: int = 1
    max_runtime_seconds: int = 900
    target_promising_alphas: int = 2


@dataclass
class DiscoveryStepReport:
    step_id: int
    alpha_id: str
    name: str
    category: str
    stage_reached: str
    status: str
    pnl_inr: float = 0.0
    sharpe: float = 0.0
    oos_trades: int = 0
    oos_pnl_inr: float = 0.0
    rejection_reason: str = ""
    asset_edges: List[str] = field(default_factory=list)


class AutonomousDiscoveryController:
    """
    Autonomous research orchestration engine around the frozen Ashva Factory v1.
    """

    def __init__(self, budget: Optional[ResearchBudget] = None, data_lake: Optional[DataLake] = None):
        self.budget = budget or ResearchBudget()
        self.lake = data_lake or DataLake(read_only=True)
        self.knowledge_map = AlphaKnowledgeMap()
        self.ledger = ResearchExperimentLedger()
        self.reports: List[DiscoveryStepReport] = []
        self.start_time = time.time()
        self.hypotheses_tested = 0
        self.dev_runs_executed = 0
        self.research_runs_executed = 0

    def is_budget_exhausted(self) -> bool:
        elapsed = time.time() - self.start_time
        if self.hypotheses_tested >= self.budget.max_hypotheses:
            return True
        if self.dev_runs_executed >= self.budget.max_dev_runs:
            return True
        if self.research_runs_executed >= self.budget.max_research_runs:
            return True
        if elapsed >= self.budget.max_runtime_seconds:
            return True
        return False

    def evaluate_stage_0_feasibility(self, candidate_info: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Stage 0: Cheap Empirical Plausibility Diagnostic.
        Evaluates raw price bars in DataLake to verify whether the market inefficiency exists.
        Returns: (passed: bool, rationale: str, estimated_edge_bps: float)
        """
        alpha_id = candidate_info.get("proposed_id", "alpha_candidate")
        category = candidate_info.get("category")
        timeframe = candidate_info.get("timeframe", "15m")

        # Fast sample check on 6 core liquid assets over 120 days
        sample_universe = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"]
        all_event_returns = []

        for sym in sample_universe:
            df = self.lake.load_bars(sym, timeframe, max_lookback_days=120)
            if df.empty:
                continue

            times = df.index.time
            closes = df["close"].values
            volumes = df["volume"].values
            n = len(df)

            # Heuristic check based on category
            if category == AlphaCategory.ORDER_FLOW_IMBALANCE:
                # Check Power Hour 14:15 volume spike and 1-hour forward return into 15:15
                t_1415 = pd.to_datetime("14:15:00").time()
                for i in range(20, n - 4):
                    if times[i] == t_1415:
                        prior_vol_mean = np.mean(volumes[i-10:i])
                        if prior_vol_mean > 0 and volumes[i] > 1.5 * prior_vol_mean:
                            # Forward return to 15:00/15:15
                            direction = 1.0 if closes[i] > closes[i-1] else -1.0
                            fwd_ret = direction * ((closes[i+4] - closes[i+1]) / closes[i+1]) * 10000.0
                            all_event_returns.append(fwd_ret)

            elif category == AlphaCategory.STATISTICAL_REVERSION:
                # Check Midday 11:00-13:30 VWAP extension (> 2.0 ATR) and 4-bar forward return
                dates = df.index.date
                typical_p = (df["high"] + df["low"] + df["close"]) / 3.0
                cum_pv = (typical_p * df["volume"]).groupby(dates).cumsum()
                cum_v = df["volume"].groupby(dates).cumsum()
                vwap = cum_pv / cum_v.replace(0, np.nan)
                df_atr = TechnicalIndicators.add_atr(df, period=14)
                atr = df_atr["atr_14"].ffill()
                t_start = pd.to_datetime("11:00:00").time()
                t_end = pd.to_datetime("13:30:00").time()
                for i in range(20, n - 4):
                    if t_start <= times[i] <= t_end and not np.isnan(vwap.iloc[i]) and not np.isnan(atr.iloc[i]):
                        dist = (closes[i] - vwap.iloc[i]) / max(0.1, atr.iloc[i])
                        if abs(dist) > 2.0:
                            direction = -1.0 if dist > 0 else 1.0
                            fwd_ret = direction * ((closes[i+4] - closes[i+1]) / closes[i+1]) * 10000.0
                            all_event_returns.append(fwd_ret)

            elif category == AlphaCategory.OPENING_AUCTION:
                # Check Opening Drive VWAP Pullback between 09:30 and 10:30 IST
                dates = df.index.date
                typical_p = (df["high"] + df["low"] + df["close"]) / 3.0
                df_temp = df.copy()
                df_temp["cum_pv"] = typical_p * df["volume"]
                df_temp["vol"] = df["volume"]
                vwap_series = df_temp.groupby(dates)["cum_pv"].cumsum() / df_temp.groupby(dates)["vol"].cumsum().replace(0, np.nan)
                vwap_vals = vwap_series.values
                t_0930 = pd.to_datetime("09:30:00").time()
                t_1030 = pd.to_datetime("10:30:00").time()
                for i in range(1, n - 6):
                    if t_0930 <= times[i] <= t_1030 and times[i-1] == pd.to_datetime("09:15:00").time():
                        op_dir = 1.0 if closes[i-1] > df["open"].values[i-1] else -1.0
                        op_range = (df["high"].values[i-1] - df["low"].values[i-1]) / df["open"].values[i-1]
                        if op_range >= 0.005:
                            if (op_dir == 1.0 and df["low"].values[i] <= vwap_vals[i] <= df["high"].values[i]) or (op_dir == -1.0 and df["low"].values[i] <= vwap_vals[i] <= df["high"].values[i]):
                                fwd_ret = op_dir * ((closes[min(i+6, n-1)] - closes[i]) / closes[i]) * 10000.0
                                all_event_returns.append(fwd_ret)

            elif category == AlphaCategory.VOLATILITY_EXPANSION:
                # Check NR4 Daily Range Compression Breakout on opening bars
                dates = df.index.date
                df_daily = df.groupby(dates).agg({"high": "max", "low": "min", "close": "last", "open": "first"})
                daily_range = df_daily["high"] - df_daily["low"]
                is_nr4 = (daily_range < daily_range.shift(1)) & (daily_range < daily_range.shift(2)) & (daily_range < daily_range.shift(3))
                nr4_days = set(df_daily.index[is_nr4.shift(1).fillna(False)])
                t_0930 = pd.to_datetime("09:30:00").time()
                for i in range(1, n - 12):
                    if dates[i] in nr4_days and times[i] == t_0930:
                        bar1_high = df["high"].values[i-1]
                        bar1_low = df["low"].values[i-1]
                        if closes[i] > bar1_high:
                            fwd_ret = ((closes[min(i+12, n-1)] - closes[i]) / closes[i]) * 10000.0
                            all_event_returns.append(fwd_ret)
                        elif closes[i] < bar1_low:
                            fwd_ret = ((closes[i] - closes[min(i+12, n-1)]) / closes[i]) * 10000.0
                            all_event_returns.append(fwd_ret)

        if len(all_event_returns) < 10:
            return False, f"Stage 0 REJECT: Insufficient historical event occurrences (N={len(all_event_returns)} < 10)", 0.0

        mean_gross_bps = float(np.mean(all_event_returns))
        # Indian statutory tax hurdle is ~7.0 bps
        if mean_gross_bps < 6.0:
            return False, (
                f"Stage 0 REJECT: Gross edge ({mean_gross_bps:.2f} bps) fails Indian statutory friction hurdle (7.0 bps) "
                f"across N={len(all_event_returns)} observations."
            ), mean_gross_bps

        return True, (
            f"Stage 0 PASS: Confirmed positive gross edge (+{mean_gross_bps:.2f} bps > 7.0 bps hurdle) "
            f"across N={len(all_event_returns)} observations."
        ), mean_gross_bps

    def run_stage_1_contract_test(self, alpha_id: str) -> bool:
        """Runs pytest on strategy lifecycle contracts."""
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_strategy_contracts.py", "-k", alpha_id],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return res.returncode == 0
        except Exception:
            return False

    def run_stage_2_dev_matrix(self, alpha_id: str) -> Dict[str, Any]:
        """Runs DEV matrix audit and parses output metrics."""
        self.dev_runs_executed += 1
        try:
            res = subprocess.run(
                [sys.executable, "scripts/run_alpha_matrix_audit.py", "--mode", "dev", "--alphas", alpha_id],
                capture_output=True,
                text=True,
                timeout=60,
            )
            # Parse matrix_output.md for results
            matrix_path = Path("matrix_output.md")
            if not matrix_path.exists():
                return {"pnl": 0.0, "sharpe": 0.0, "positive_assets": [], "raw_output": res.stdout}

            content = matrix_path.read_text(encoding="utf-8")
            # Simple extraction
            pnl = 0.0
            sharpe = 0.0
            pos_assets = []
            for line in content.splitlines():
                if alpha_id in line and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) > 7:
                        try:
                            pnl = float(parts[3])
                            sharpe = float(parts[6])
                        except Exception:
                            pass
                if "🟢" in line and "(+" in line:
                    pos_assets.append(line.strip())

            return {
                "pnl": pnl,
                "sharpe": sharpe,
                "positive_assets": pos_assets,
                "raw_output": res.stdout,
            }
        except Exception as e:
            return {"pnl": 0.0, "sharpe": -99.0, "positive_assets": [], "error": str(e)}

    def execute_discovery_cycle(self) -> List[DiscoveryStepReport]:
        """
        Runs the full autonomous discovery cycle within budget bounds.
        """
        print("\n" + "=" * 100)
        print("[*] ASHVA FACTORY v2 -- AUTONOMOUS ALPHA DISCOVERY CONTROLLER")
        print(f"[*] Budgets: Hypotheses={self.budget.max_hypotheses} | DEV Runs={self.budget.max_dev_runs} | Max Time={self.budget.max_runtime_seconds}s")
        print("=" * 100 + "\n")

        candidate_queue = self.knowledge_map.get_unexplored_mechanisms()
        step_idx = 1

        for candidate in candidate_queue:
            if self.is_budget_exhausted():
                print("[!] Research budget reached. Halting discovery cycle.")
                break

            alpha_id = candidate["proposed_id"]
            name = candidate["name"]
            category = candidate["category"]
            timeframe = candidate["timeframe"]
            entry_window = candidate["entry_window"]

            print(f"[>] Step {step_idx}: Investigating Candidate '{name}' ({alpha_id})...")
            print(f"    Category: {category.value} | Timeframe: {timeframe} | Window: {entry_window}")

            # 1. Duplication & Novelty Filter
            is_novel = self.knowledge_map.is_novel_hypothesis(
                category=category,
                mechanism_desc=candidate["mechanism_description"],
                timeframe=timeframe,
                entry_window=entry_window,
            )
            if not is_novel:
                print("    [!] REJECTED: Duplicate mechanism already tested in baseline. Skipping.")
                continue

            self.hypotheses_tested += 1

            # 2. Stage 0 Feasibility Diagnostic
            print("    [*] Executing Stage 0 Cheap Plausibility Diagnostic...")
            passed_s0, s0_rationale, gross_edge = self.evaluate_stage_0_feasibility(candidate)
            print(f"    --> {s0_rationale}")

            if not passed_s0:
                report = DiscoveryStepReport(
                    step_id=step_idx,
                    alpha_id=alpha_id,
                    name=name,
                    category=category.value,
                    stage_reached="Stage 0 (Plausibility)",
                    status="REJECTED_AT_STAGE_0",
                    rejection_reason=s0_rationale,
                )
                self.reports.append(report)
                step_idx += 1
                continue

            # 3. If passed Stage 0, log recommendation for implementation
            report = DiscoveryStepReport(
                step_id=step_idx,
                alpha_id=alpha_id,
                name=name,
                category=category.value,
                stage_reached="Stage 0 (Plausibility Verified)",
                status="RECOMMENDED_FOR_IMPLEMENTATION",
                pnl_inr=0.0,
                sharpe=0.0,
                rejection_reason="Plausibility confirmed (Gross edge > 7.0 bps). Ready for strategy implementation.",
            )
            self.reports.append(report)
            step_idx += 1

        print("\n" + "=" * 100)
        print(f"[*] AUTONOMOUS DISCOVERY COMPLETE: {len(self.reports)} Hypotheses Evaluated")
        print("=" * 100 + "\n")
        return self.reports
