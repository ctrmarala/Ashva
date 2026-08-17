"""
Ashva Alpha Decay & Concept Drift Detector
Continuously evaluates live trade performance against historical backtested distributions
using Kolmogorov-Smirnov 2-sample tests to flag decaying alpha and concept drift.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import ks_2samp


class AlphaDriftDetector:
    """
    Statistical watchdog to monitor alpha decay and distribution drift.
    """

    def __init__(self, ks_alpha_threshold: float = 0.05, min_samples_required: int = 15):
        self.ks_alpha_threshold = ks_alpha_threshold
        self.min_samples_required = min_samples_required

    def evaluate_drift(
        self,
        backtest_returns: np.ndarray,
        live_returns: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Runs Kolmogorov-Smirnov 2-sample test comparing live return distribution to backtest.
        
        :return: Dict containing drift status, KS-statistic, p-value, and recommendation.
        """
        clean_bt = backtest_returns[~np.isnan(backtest_returns)]
        clean_live = live_returns[~np.isnan(live_returns)]

        if len(clean_live) < self.min_samples_required or len(clean_bt) < self.min_samples_required:
            return {
                "status": "INSUFFICIENT_DATA",
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "live_samples": len(clean_live),
                "message": f"Need at least {self.min_samples_required} live trades to evaluate drift.",
            }

        # Run 2-sample Kolmogorov-Smirnov test
        ks_stat, p_val = ks_2samp(clean_bt, clean_live)

        # Check performance decay (Mean comparison)
        bt_mean = float(np.mean(clean_bt))
        live_mean = float(np.mean(clean_live))

        has_drift = p_val < self.ks_alpha_threshold
        performance_decay = live_mean < 0 and live_mean < bt_mean

        if has_drift and performance_decay:
            status = "ALPHA_DECAY_HALT"
            msg = f"Severe Alpha Decay: Return distribution shifted (p={p_val:.4f}) and live mean is negative ({live_mean*100:.2f}%)."
        elif has_drift:
            status = "WARNING_DRIFT"
            msg = f"Distribution Drift Detected (p={p_val:.4f}): Market regime has diverged from backtested sample."
        else:
            status = "STABLE"
            msg = f"Alpha is Stable: Live return distribution matches backtest expectations (p={p_val:.4f})."

        return {
            "status": status,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "backtest_mean_return_pct": round(bt_mean * 100.0, 3),
            "live_mean_return_pct": round(live_mean * 100.0, 3),
            "live_samples": len(clean_live),
            "message": msg,
        }
