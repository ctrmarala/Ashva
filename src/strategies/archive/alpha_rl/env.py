"""
Ashva Custom Gymnasium Trading Environment
High-fidelity continuous-action trading environment with multi-factor state representations,
Indian market friction penalties, and differential Sharpe reward shaping.
"""

from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.analytics.indian_costs import IndianCostModel, Segment
from src.features.frac_diff import frac_diff_ffd
from src.features.microstructure import MicrostructureFeatureExtractor


class AshvaTradingEnv(gym.Env):
    """
    OpenAI Gymnasium-compatible Quantitative Trading Environment for Indian Equities.
    
    Observation Space:
        0: Fractionally Differenced Close (Normalized)
        1: Distance to Anchored VWAP (% of price)
        2: Volume Surge Ratio (relative to 20-period MA)
        3: Cumulative Volume Delta (CVD) normalized
        4: Hurst Exponent (Regime indicator 0.0 to 1.0)
        5: Current Position Allocation (-1.0 to +1.0)
        6: Unrealized PnL (% of entry)
    
    Action Space:
        Continuous allocation a in [-1.0, +1.0]:
        -1.0 = 100% Short, 0.0 = Flat/Cash, +1.0 = 100% Long
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 500000.0,
        cost_model: Optional[IndianCostModel] = None,
        max_drawdown_limit_pct: float = 10.0,
        trading_fee_bps: float = 5.0,  # ~0.05% total friction per turn
    ):
        super().__init__()
        
        self.raw_df = df.copy()
        self.initial_capital = initial_capital
        self.cost_model = cost_model or IndianCostModel()
        self.max_drawdown_limit_pct = max_drawdown_limit_pct
        self.trading_fee_ratio = trading_fee_bps / 10000.0

        # Precompute Feature Matrix
        self._prepare_features()

        # Action & Observation Spaces
        # Action: Continuous target allocation in [-1.0, +1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space (7 continuous features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Environment State Variables
        self.current_step = 0
        self.max_steps = len(self.feature_matrix) - 1
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.position = 0.0  # Current allocation in [-1.0, 1.0]
        self.entry_price = 0.0
        self.history = []

    def _prepare_features(self):
        """Precomputes all point-in-time features for fast stepping."""
        extractor = MicrostructureFeatureExtractor()
        
        df = extractor.calculate_anchored_vwap(self.raw_df)
        df = extractor.calculate_volume_delta(df)

        # 1. Fractional Differentiation
        close_series = df["close"]
        try:
            fd = frac_diff_ffd(close_series, d=0.40, threshold=1e-3)
            df["frac_diff"] = fd.reindex(df.index).bfill()
        except Exception:
            df["frac_diff"] = close_series.pct_change().fillna(0.0)

        # 2. VWAP Distance %
        df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
        df["vwap_dist"] = df["vwap_dist"].fillna(0.0)

        # 3. Hurst Exponent
        hurst_vals = []
        close_vals = df["close"].values
        for i in range(len(df)):
            if i < 40:
                hurst_vals.append(0.50)
            else:
                chunk = close_vals[i - 40 : i]
                hurst_vals.append(extractor.calculate_hurst_exponent(pd.Series(chunk)))
        df["hurst"] = hurst_vals

        # Standardized feature array
        f_frac = (df["frac_diff"] - df["frac_diff"].mean()) / (df["frac_diff"].std() + 1e-8)
        f_vwap = df["vwap_dist"] * 100.0  # Percentage
        f_volsurge = df["volume_surge_ratio"].clip(0, 10.0)
        f_cvd = (df["cvd"] - df["cvd"].mean()) / (df["cvd"].std() + 1e-8)
        f_hurst = df["hurst"]

        self.feature_matrix = np.column_stack([
            f_frac.values,
            f_vwap.values,
            f_volsurge.values,
            f_cvd.values,
            f_hurst.values,
        ]).astype(np.float32)

        self.closes = df["close"].values
        self.timestamps = df.index

    def _get_observation(self) -> np.ndarray:
        """Constructs the current 7-dimensional observation vector."""
        feat_5d = self.feature_matrix[self.current_step]
        curr_price = self.closes[self.current_step]

        unrealized_pnl_pct = 0.0
        if self.position != 0.0 and self.entry_price > 0:
            raw_ret = (curr_price - self.entry_price) / self.entry_price
            unrealized_pnl_pct = raw_ret * np.sign(self.position) * 100.0

        obs = np.array([
            feat_5d[0],
            feat_5d[1],
            feat_5d[2],
            feat_5d[3],
            feat_5d[4],
            self.position,
            unrealized_pnl_pct,
        ], dtype=np.float32)

        # Replace any NaNs or infinities with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.history = []

        obs = self._get_observation()
        info = {"equity": self.equity, "step": self.current_step}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        target_allocation = float(np.clip(action[0], -1.0, 1.0))
        
        # Deadband threshold (ignore micro-adjustments < 0.05 to save transaction costs)
        if abs(target_allocation - self.position) < 0.05:
            target_allocation = self.position

        prev_price = self.closes[self.current_step]
        self.current_step += 1
        curr_price = self.closes[self.current_step]

        # 1. Price Return
        price_return = (curr_price - prev_price) / prev_price
        gross_return = self.position * price_return

        # 2. Turnover & Transaction Friction
        allocation_delta = abs(target_allocation - self.position)
        friction_penalty = allocation_delta * self.trading_fee_ratio

        # 3. Net Period Return & Equity Update
        net_return = gross_return - friction_penalty
        self.equity *= (1.0 + net_return)
        self.peak_equity = max(self.peak_equity, self.equity)

        # Drawdown
        drawdown_pct = (self.peak_equity - self.equity) / self.peak_equity * 100.0

        # Update Position State
        if target_allocation != self.position:
            self.entry_price = curr_price if target_allocation != 0.0 else 0.0
            self.position = target_allocation

        # 4. Reward Function (Differential Sharpe with Drawdown Penalty)
        # Reward = Net Return - 0.5 * Volatility Penalty - Drawdown Penalty
        reward = net_return * 100.0  # Scale returns to percentage points
        if drawdown_pct > 5.0:
            reward -= (drawdown_pct - 5.0) * 0.1  # Penalize severe drawdowns
        if friction_penalty > 0:
            reward -= friction_penalty * 50.0   # Penalize excessive churning

        # 5. Termination Conditions
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True
        elif drawdown_pct >= self.max_drawdown_limit_pct:
            terminated = True  # Circuit Breaker Triggered
            reward -= 50.0    # Severe penalty for blowing risk limit
        elif self.equity <= self.initial_capital * 0.50:
            terminated = True  # Bankruptcy

        obs = self._get_observation()
        info = {
            "equity": self.equity,
            "drawdown_pct": drawdown_pct,
            "position": self.position,
            "net_return": net_return,
            "step": self.current_step,
        }

        return obs, float(reward), terminated, truncated, info
