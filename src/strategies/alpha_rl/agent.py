"""
Ashva Deep Reinforcement Learning Agent (PPO / Actor-Critic)
Continuous policy gradient agent for dynamic multi-factor portfolio allocation.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata
from src.strategies.alpha_rl.env import AshvaTradingEnv
from src.core.events import BarEvent, SignalEvent


class PolicyNetwork:
    """
    Two-layer MLP Parameterizing Gaussian Policy: Mean mu(s) and Log-Std log_std(s).
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32, seed: int = 42):
        np.random.seed(seed)
        # Xavier / He Initialization
        self.w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        # Policy Mean Output
        self.w_mu = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_mu = np.zeros(1)
        
        # Policy Value Function Baseline (Critic)
        self.w_val = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_val = np.zeros(1)

        self.log_std = np.zeros(1) - 0.5  # Initial std ~ 0.60

    def forward(self, state: np.ndarray) -> Tuple[float, float, float]:
        """
        Forward pass computing (action_mean, action_std, value_estimate).
        """
        # Hidden Layer with ReLU
        h = np.maximum(0.0, np.dot(state, self.w1) + self.b1)
        
        # Action Mean mapped to [-1.0, 1.0] via tanh
        mu = float(np.tanh(np.dot(h, self.w_mu) + self.b_mu)[0])
        std = float(np.exp(np.clip(self.log_std[0], -2.0, 1.0)))
        
        # Critic Value Estimate
        val = float((np.dot(h, self.w_val) + self.b_val)[0])
        return mu, std, val

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float, float]:
        mu, std, val = self.forward(state)
        if deterministic:
            return float(np.clip(mu, -1.0, 1.0)), 0.0, val
        
        action = np.random.normal(mu, std)
        clipped_action = float(np.clip(action, -1.0, 1.0))
        
        # Log probability of Gaussian
        var = std ** 2
        log_prob = -0.5 * (((clipped_action - mu) ** 2) / (var + 1e-8) + np.log(2.0 * np.pi * var + 1e-8))
        return clipped_action, float(log_prob), val


class AlphaRLAgent(BaseStrategy, BaseHypothesis):
    """
    Hypothesis 4: Autonomous Deep Reinforcement Learning Agent.
    """

    DEFAULT_METADATA = HypothesisMetadata(
        hypothesis_id="ALPHA_04_DEEP_RL_AGENT",
        name="Deep Reinforcement Learning Dynamic Multi-Factor Allocation Agent",
        category="REINFORCEMENT_LEARNING",
        economic_rationale=(
            "Non-linear interactions between fractional price memory, order flow imbalances, "
            "and market regimes are dynamically mapped to optimal exposure weights via policy gradients."
        ),
        target_instruments=["RELIANCE", "NIFTYBEES", "HDFCBANK", "INFY", "TCS"],
        timeframe="5m",
    )

    def __init__(
        self,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or self.DEFAULT_METADATA
        BaseHypothesis.__init__(self, metadata=meta, parameters=parameters)
        BaseStrategy.__init__(self, strategy_id=meta.hypothesis_id, parameters=parameters)

        self.policy = PolicyNetwork(input_dim=7, hidden_dim=32)
        self.trained = False

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {"hidden_dim": [16, 32, 64]}

    def train_on_env(
        self,
        env: AshvaTradingEnv,
        num_episodes: int = 15,
        learning_rate: float = 0.005,
        gamma: float = 0.99,
    ) -> List[float]:
        """
        Trains the policy network using Actor-Critic Policy Gradients.
        """
        episode_rewards = []

        for ep in range(num_episodes):
            state, _ = env.reset()
            states, actions, rewards, values, log_probs = [], [], [], [], []
            done = False
            total_reward = 0.0

            while not done:
                action, log_prob, val = self.policy.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(np.array([action]))
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(val)
                log_probs.append(log_prob)

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)

            # Compute Discounted Cumulative Returns
            discounted_returns = []
            running_add = 0.0
            for r in reversed(rewards):
                running_add = r + gamma * running_add
                discounted_returns.insert(0, running_add)

            # Compute Advantages: A(s, a) = G_t - V(s)
            returns_arr = np.array(discounted_returns)
            values_arr = np.array(values)
            advantages = returns_arr - values_arr
            adv_std = np.std(advantages)
            if adv_std > 1e-8:
                advantages = (advantages - np.mean(advantages)) / adv_std

            # Simple Policy Gradient Parameter Update
            for i in range(len(states)):
                s = states[i]
                a = actions[i]
                adv = advantages[i]
                
                h = np.maximum(0.0, np.dot(s, self.policy.w1) + self.policy.b1)
                mu, std, _ = self.policy.forward(s)
                
                # Grad log pi / grad mu = (a - mu) / std^2
                d_mu = (a - mu) / (std ** 2 + 1e-8)
                
                # Policy gradient step
                grad_w_mu = np.outer(h, d_mu * adv)
                self.policy.w_mu += learning_rate * np.clip(grad_w_mu, -1.0, 1.0)
                
                # Critic value step (MSE loss)
                val_err = returns_arr[i] - values_arr[i]
                grad_w_val = np.outer(h, val_err)
                self.policy.w_val += learning_rate * np.clip(grad_w_val, -1.0, 1.0)

        self.trained = True
        return episode_rewards

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes trained RL policy over historical DataFrame to generate signal series.
        """
        env = AshvaTradingEnv(df=df)
        obs, _ = env.reset()
        signals = [0.0]

        for _ in range(len(df) - 1):
            action, _, _ = self.policy.get_action(obs, deterministic=True)
            signals.append(action)
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            if terminated or truncated:
                break

        # Align length
        if len(signals) < len(df):
            signals.extend([0.0] * (len(df) - len(signals)))

        df_out = df.copy()
        df_out["signal"] = signals[: len(df)]
        return df_out

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        return None
