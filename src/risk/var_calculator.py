"""
Ashva Value at Risk (VaR) & Expected Shortfall (CVaR) Engine
Computes Parametric Cornish-Fisher VaR, Historical VaR, and Conditional Value at Risk (Expected Shortfall).
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


class RiskMetricsCalculator:
    """
    Computes real-time and multi-day Value at Risk (VaR) and CVaR.
    """

    @staticmethod
    def calculate_parametric_var(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        portfolio_value: float = 500000.0,
    ) -> Dict[str, float]:
        """
        Computes Gaussian VaR and Cornish-Fisher VaR (adjusted for non-normality).
        """
        clean_ret = returns[~np.isnan(returns)]
        if len(clean_ret) < 10:
            return {"gaussian_var_pct": 0.0, "cornish_fisher_var_pct": 0.0, "var_inr": 0.0}

        mu = float(np.mean(clean_ret))
        sigma = float(np.std(clean_ret, ddof=1))
        z = float(norm.ppf(confidence_level))

        # 1. Standard Gaussian VaR
        gaussian_var = -(mu - z * sigma)

        # 2. Cornish-Fisher Expansion (Adjusts for Skewness and Kurtosis)
        s = float(skew(clean_ret))
        k = float(kurtosis(clean_ret, fisher=True))  # Excess kurtosis (normal = 0)

        z_cf = z + (z**2 - 1) * s / 6.0 + (z**3 - 3 * z) * k / 24.0 - (2 * z**3 - 5 * z) * (s**2) / 36.0
        cf_var = -(mu - z_cf * sigma)

        var_pct = max(0.0, float(cf_var)) * 100.0
        var_inr = (var_pct / 100.0) * portfolio_value

        return {
            "gaussian_var_pct": round(max(0.0, float(gaussian_var)) * 100.0, 3),
            "cornish_fisher_var_pct": round(var_pct, 3),
            "var_inr": round(var_inr, 2),
            "confidence_level": confidence_level,
        }

    @staticmethod
    def calculate_historical_var_cvar(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        portfolio_value: float = 500000.0,
    ) -> Dict[str, float]:
        """
        Computes Historical Empirical VaR and Conditional VaR (Expected Shortfall / CVaR).
        """
        clean_ret = returns[~np.isnan(returns)]
        if len(clean_ret) < 10:
            return {"historical_var_pct": 0.0, "cvar_expected_shortfall_pct": 0.0, "cvar_inr": 0.0}

        alpha = (1.0 - confidence_level) * 100.0
        var_threshold = float(np.percentile(clean_ret, alpha))
        var_pct = max(0.0, -var_threshold) * 100.0

        # CVaR: Mean of all returns worse than VaR threshold
        tail_losses = clean_ret[clean_ret <= var_threshold]
        cvar_threshold = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_threshold
        cvar_pct = max(0.0, -cvar_threshold) * 100.0
        cvar_inr = (cvar_pct / 100.0) * portfolio_value

        return {
            "historical_var_pct": round(var_pct, 3),
            "cvar_expected_shortfall_pct": round(cvar_pct, 3),
            "cvar_inr": round(cvar_inr, 2),
            "confidence_level": confidence_level,
        }
