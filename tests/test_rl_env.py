"""
Unit Tests for Custom Gymnasium Trading Environment and RL Policy
"""

import numpy as np
import pandas as pd
import pytest
from src.strategies.alpha_rl.env import AshvaTradingEnv
from src.strategies.alpha_rl.agent import AlphaRLAgent, PolicyNetwork


@pytest.fixture
def mock_ohlcv_series():
    dates = pd.date_range("2026-01-01 09:15", periods=60, freq="5min")
    np.random.seed(42)
    prices = 2500.0 + np.cumsum(np.random.normal(0, 2.0, 60))
    df = pd.DataFrame({
        "open": prices - 1.0,
        "high": prices + 2.0,
        "low": prices - 2.0,
        "close": prices,
        "volume": np.random.randint(1000, 5000, 60),
    }, index=dates)
    return df


def test_trading_env_lifecycle(mock_ohlcv_series):
    env = AshvaTradingEnv(df=mock_ohlcv_series, initial_capital=100000.0)
    
    # 1. Reset
    obs, info = env.reset()
    assert obs.shape == (7,)
    assert info["equity"] == 100000.0
    assert not np.isnan(obs).any()

    # 2. Step with Long Action
    next_obs, reward, terminated, truncated, step_info = env.step(np.array([1.0], dtype=np.float32))
    assert next_obs.shape == (7,)
    assert isinstance(reward, float)
    assert step_info["position"] == 1.0
    assert step_info["step"] == 1


def test_policy_network_forward_and_action():
    policy = PolicyNetwork(input_dim=7, hidden_dim=16)
    dummy_state = np.zeros(7, dtype=np.float32)

    mu, std, val = policy.forward(dummy_state)
    assert -1.0 <= mu <= 1.0
    assert std > 0.0

    action, log_prob, _ = policy.get_action(dummy_state)
    assert -1.0 <= action <= 1.0
    assert isinstance(log_prob, float)


def test_rl_agent_training_loop(mock_ohlcv_series):
    env = AshvaTradingEnv(df=mock_ohlcv_series)
    agent = AlphaRLAgent()

    # Run quick 3-episode training
    rewards = agent.train_on_env(env, num_episodes=3)
    assert len(rewards) == 3
    assert agent.trained is True

    # Generate signals
    signals_df = agent.generate_signals(mock_ohlcv_series)
    assert "signal" in signals_df.columns
    assert len(signals_df) == len(mock_ohlcv_series)
