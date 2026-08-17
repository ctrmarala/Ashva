"""
Unit Tests for Alpha Decay & Concept Drift Detector
"""

import numpy as np
import pytest
from src.analytics.drift_detector import AlphaDriftDetector


def test_drift_detector():
    detector = AlphaDriftDetector(min_samples_required=20)
    np.random.seed(42)

    # 1. Backtest distribution (Mean = +0.5%)
    bt_returns = np.random.normal(loc=0.005, scale=0.01, size=100)

    # 2. Live returns from exact same distribution -> STABLE
    live_stable = np.random.normal(loc=0.005, scale=0.01, size=50)
    res_stable = detector.evaluate_drift(bt_returns, live_stable)
    assert res_stable["status"] == "STABLE"
    assert res_stable["p_value"] > 0.05

    # 3. Live returns from heavily shifted negative distribution -> ALPHA_DECAY_HALT
    live_decay = np.random.normal(loc=-0.015, scale=0.01, size=50)
    res_decay = detector.evaluate_drift(bt_returns, live_decay)
    assert res_decay["status"] == "ALPHA_DECAY_HALT"
    assert res_decay["p_value"] < 0.05
