"""
Ashva Deep Reinforcement Learning Agent Training CLI
Trains continuous Actor-Critic / PPO policy network on AshvaTradingEnv with Indian market friction.

Usage:
    python scripts/run_rl_train.py --symbol RELIANCE --timeframe 5m --episodes 15
"""

import argparse
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_lake import DataLake
from src.strategies.alpha_rl.env import AshvaTradingEnv
from src.strategies.alpha_rl.agent import AlphaRLAgent
from src.backtest.engine import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="Ashva RL Training Pipeline")
    parser.add_argument("--symbol", type=str, default="RELIANCE", help="Symbol to train on")
    parser.add_argument("--timeframe", type=str, default="5m", help="Candle timeframe")
    parser.add_argument("--episodes", type=int, default=15, help="Number of training episodes")

    args = parser.parse_args()
    data_lake = DataLake()

    print("=" * 80)
    print(f"[*] ASHVA DEEP REINFORCEMENT LEARNING TRAINING PIPELINE")
    print(f"[*] Target Asset: {args.symbol} | Timeframe: {args.timeframe} | Training Episodes: {args.episodes}")
    print("=" * 80)

    # 1. Load Data
    df = data_lake.load_bars(args.symbol, args.timeframe)
    if df.empty:
        print(f"[-] No market data found for {args.symbol} in DataLake. Please sync via Angel One SmartAPI first.")
        return

    print(f"[+] Loaded {len(df)} candles for RL Environment.\n")

    # 2. Initialize Gymnasium Environment & Agent
    env = AshvaTradingEnv(df=df, initial_capital=500000.0)
    agent = AlphaRLAgent()

    print("[*] Training Actor-Critic Policy Network...")
    episode_rewards = agent.train_on_env(env=env, num_episodes=args.episodes)

    for ep, reward in enumerate(episode_rewards):
        print(f"  Episode {ep+1:02d}/{args.episodes:02d} - Total Reward: {reward:+.2f}")

    print("\n[+] Policy training completed successfully!")

    # 3. Evaluate Agent in Backtest Engine
    print("\n" + "=" * 80)
    print("[*] EVALUATING TRAINED RL AGENT IN INSTITUTIONAL BACKTEST")
    print("=" * 80)
    signals_df = agent.generate_signals(df)
    
    engine = BacktestEngine(initial_capital=500000.0)
    bt_result = engine.run(signals_df, symbol=args.symbol, strategy_id=agent.strategy_id)
    
    for k, v in bt_result.summary().items():
        print(f"  {k:28s}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
