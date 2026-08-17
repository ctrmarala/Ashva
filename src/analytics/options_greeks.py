"""
Ashva Quantitative Options & Derivatives Analytics Engine
Calculates analytical Black-Scholes-Merton option pricing, Implied Volatility (IV) inversion,
and First/Second-Order Greeks (Delta, Gamma, Theta, Vega, Rho) for NSE NIFTY and BANKNIFTY options.
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class BlackScholesGreeks:
    """
    Institutional Options Pricing and Greeks Engine.
    """

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
        """
        Calculates d1 and d2 for Black-Scholes formula.
        S: Spot Price
        K: Strike Price
        T: Time to Expiration in Years
        r: Risk-free rate (e.g. 0.065 for RBI 6.5% repo rate)
        sigma: Annualized Volatility
        """
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(d1), float(d2)

    @classmethod
    def calculate_price(
        cls,
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        volatility: float,
        risk_free_rate: float = 0.065,
        option_type: str = "CE",  # "CE" (Call) or "PE" (Put)
    ) -> float:
        """Calculates theoretical option premium using Black-Scholes-Merton formula."""
        if time_to_expiry_years <= 0:
            if option_type.upper() == "CE":
                return max(0.0, spot - strike)
            else:
                return max(0.0, strike - spot)

        d1, d2 = cls._d1_d2(spot, strike, time_to_expiry_years, risk_free_rate, volatility)
        
        if option_type.upper() == "CE":
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry_years) * norm.cdf(d2)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_expiry_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return max(0.05, float(price))

    @classmethod
    def calculate_implied_volatility(
        cls,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        risk_free_rate: float = 0.065,
        option_type: str = "CE",
    ) -> float:
        """Inverts Black-Scholes price to compute Implied Volatility (IV) via Brent's method."""
        if time_to_expiry_years <= 0 or market_price <= 0:
            return 0.15  # Fallback 15% IV

        def objective(sigma):
            return cls.calculate_price(spot, strike, time_to_expiry_years, sigma, risk_free_rate, option_type) - market_price

        try:
            iv = brentq(objective, a=0.001, b=5.0, xtol=1e-5, maxiter=100)
            return float(iv)
        except Exception:
            return 0.15

    @classmethod
    def calculate_greeks(
        cls,
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        volatility: float,
        risk_free_rate: float = 0.065,
        option_type: str = "CE",
    ) -> Dict[str, float]:
        """
        Computes all first and second order Greeks:
        - Delta: Change in option price per 1 point spot move
        - Gamma: Change in delta per 1 point spot move
        - Theta: Daily time decay in rupees/points
        - Vega: Change in option price per 1% change in volatility
        - Rho: Change in option price per 1% change in interest rates
        """
        if time_to_expiry_years <= 0.0001:
            is_ce = option_type.upper() == "CE"
            delta = 1.0 if (is_ce and spot > strike) else (-1.0 if (not is_ce and spot < strike) else 0.0)
            return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

        d1, d2 = cls._d1_d2(spot, strike, time_to_expiry_years, risk_free_rate, volatility)
        pdf_d1 = norm.pdf(d1)
        sqrt_t = np.sqrt(time_to_expiry_years)
        discount = np.exp(-risk_free_rate * time_to_expiry_years)

        # Gamma (Same for Call and Put)
        gamma = pdf_d1 / (spot * volatility * sqrt_t) if (spot * volatility * sqrt_t) > 0 else 0.0

        # Vega (Same for Call and Put, normalized per 1% change in vol)
        vega = (spot * pdf_d1 * sqrt_t) / 100.0

        if option_type.upper() == "CE":
            delta = norm.cdf(d1)
            theta_annual = -(spot * pdf_d1 * volatility) / (2 * sqrt_t) - risk_free_rate * strike * discount * norm.cdf(d2)
            rho = (strike * time_to_expiry_years * discount * norm.cdf(d2)) / 100.0
        else:
            delta = norm.cdf(d1) - 1.0
            theta_annual = -(spot * pdf_d1 * volatility) / (2 * sqrt_t) + risk_free_rate * strike * discount * norm.cdf(-d2)
            rho = -(strike * time_to_expiry_years * discount * norm.cdf(-d2)) / 100.0

        # Theta normalized per calendar day (365 days)
        theta_daily = theta_annual / 365.0

        return {
            "delta": round(float(delta), 4),
            "gamma": round(float(gamma), 6),
            "theta_daily": round(float(theta_daily), 4),
            "vega": round(float(vega), 4),
            "rho": round(float(rho), 4),
            "theoretical_price": round(cls.calculate_price(spot, strike, time_to_expiry_years, volatility, risk_free_rate, option_type), 2),
        }
