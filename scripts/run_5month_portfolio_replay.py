"""
Ashva 5-Month Portfolio Replay Runner (Forwarder to Unified Portfolio Replay)
"""
import sys
from pathlib import Path

ROOT_DIR = Path(r"c:\Work\Ashva")
sys.path.insert(0, str(ROOT_DIR))

from scripts.run_unified_portfolio_replay import main

if __name__ == "__main__":
    main()
