"""
Ashva Research Feedback Loop Analyzer
Scans executed trading ledger records (Replay, Paper, Live) and generates structured quantitative hypotheses
for the Alpha Factory (e.g., time-of-day decay, regime shifts, slippage drag, MFE capture degradation).
MANDATE: Emits research hypotheses to Factory backlog only; NEVER automatically modifies active alphas.
"""

from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from src.trading.ledger import TradingLedger

logger = logging.getLogger("Ashva.ResearchFeedback")


class ResearchFeedbackAnalyzer:
    """
    Translates execution telemetry into quantitative research hypotheses.
    """

    def __init__(self, ledger: Optional[TradingLedger] = None):
        self.ledger = ledger or TradingLedger()

    def analyze_alpha_performance(self, alpha_id: str) -> List[Dict[str, Any]]:
        """
        Analyzes an alpha's closed trades from the TradingLedger and emits structured research hypotheses.
        """
        trades = self.ledger.get_trades(alpha_id=alpha_id, limit=500)
        if not trades or len(trades) < 5:
            return []

        df = pd.DataFrame(trades)
        hypotheses = []

        # 1. MFE Capture Efficiency Analysis
        # MFE capture = Net PnL / MFE (if MFE > 0)
        if "mfe" in df.columns and "net_pnl" in df.columns:
            positive_mfe = df[df["mfe"] > 0]
            if not positive_mfe.empty:
                capture_ratios = positive_mfe["net_pnl"] / positive_mfe["mfe"]
                mean_capture = float(capture_ratios.mean())

                if mean_capture < 0.35:
                    hypotheses.append({
                        "hypothesis_type": "TRAILING_STOP_OPTIMIZATION",
                        "alpha_id": alpha_id,
                        "title": f"Suboptimal MFE Capture Ratio ({mean_capture*100:.1f}%) for {alpha_id}",
                        "observation": f"Strategy realizes only {mean_capture*100:.1f}% of its peak favorable excursion before exit.",
                        "suggested_action": "Test dynamic ATR ratchet trailing stops or tighter partial profit-taking in Alpha Factory.",
                        "priority": "HIGH",
                        "generated_at": datetime.now().isoformat(),
                    })

        # 2. Time-of-Day Performance Decay
        if "entry_time" in df.columns:
            df["entry_dt"] = pd.to_datetime(df["entry_time"])
            df["entry_hour"] = df["entry_dt"].dt.hour
            morning = df[df["entry_hour"] <= 10]
            afternoon = df[df["entry_hour"] >= 13]

            if not morning.empty and not afternoon.empty:
                morning_win_rate = (morning["net_pnl"] > 0).mean() * 100.0
                afternoon_win_rate = (afternoon["net_pnl"] > 0).mean() * 100.0

                if (morning_win_rate - afternoon_win_rate) > 25.0:
                    hypotheses.append({
                        "hypothesis_type": "TIME_WINDOW_REDUCTION",
                        "alpha_id": alpha_id,
                        "title": f"Afternoon Edge Decay for {alpha_id} (Morning: {morning_win_rate:.0f}% vs Afternoon: {afternoon_win_rate:.0f}%)",
                        "observation": "Alpha performance deteriorates significantly after 13:00 IST.",
                        "suggested_action": "Restrict entry window in QualifiedAlphaContract from 09:30-15:00 to 09:30-11:30.",
                        "priority": "MEDIUM",
                        "generated_at": datetime.now().isoformat(),
                    })

        # 3. Slippage & Cost Impact
        if "slippage_paid" in df.columns and "total_costs" in df.columns:
            total_net = df["net_pnl"].sum()
            total_costs = df["total_costs"].sum()
            if total_net > 0 and (total_costs / total_net) > 0.40:
                hypotheses.append({
                    "hypothesis_type": "TURNOVER_COST_FRICTION",
                    "alpha_id": alpha_id,
                    "title": f"High Frictions Burden on {alpha_id} (Costs = {total_costs/total_net*100:.1f}% of Net PnL)",
                    "observation": f"Exchange charges and slippage consume {total_costs/total_net*100:.1f}% of net profits.",
                    "suggested_action": "Increase minimum target profit hurdle or test higher timeframes (e.g. 15m -> 30m).",
                    "priority": "HIGH",
                    "generated_at": datetime.now().isoformat(),
                })

        return hypotheses
