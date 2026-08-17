"""
Unit Tests for Options Pricing, Implied Volatility, and Greeks Engine
"""

import pytest
from src.analytics.options_greeks import BlackScholesGreeks


def test_black_scholes_pricing_and_greeks():
    # NIFTY 50 @ 24,000, Strike 24,000 (ATM), 7 days to expiry (7/365 years), IV = 15%, r = 6.5%
    spot = 24000.0
    strike = 24000.0
    t_exp = 7.0 / 365.0
    sigma = 0.15
    r = 0.065

    # 1. Call Option
    ce_price = BlackScholesGreeks.calculate_price(spot, strike, t_exp, sigma, r, option_type="CE")
    ce_greeks = BlackScholesGreeks.calculate_greeks(spot, strike, t_exp, sigma, r, option_type="CE")

    assert ce_price > 100.0
    assert 0.45 <= ce_greeks["delta"] <= 0.60  # ATM Call delta ~ 0.50
    assert ce_greeks["gamma"] > 0.0
    assert ce_greeks["theta_daily"] < 0.0  # Daily time decay is negative for option buyers
    assert ce_greeks["vega"] > 0.0

    # 2. Put Option
    pe_price = BlackScholesGreeks.calculate_price(spot, strike, t_exp, sigma, r, option_type="PE")
    pe_greeks = BlackScholesGreeks.calculate_greeks(spot, strike, t_exp, sigma, r, option_type="PE")

    assert pe_price > 100.0
    assert -0.60 <= pe_greeks["delta"] <= -0.45  # ATM Put delta ~ -0.50
    assert pe_greeks["theta_daily"] < 0.0


def test_implied_volatility_solver():
    spot = 24000.0
    strike = 24000.0
    t_exp = 7.0 / 365.0
    true_iv = 0.18

    market_price = BlackScholesGreeks.calculate_price(spot, strike, t_exp, true_iv, 0.065, "CE")
    solved_iv = BlackScholesGreeks.calculate_implied_volatility(market_price, spot, strike, t_exp, 0.065, "CE")

    assert pytest.approx(solved_iv, 0.01) == true_iv
