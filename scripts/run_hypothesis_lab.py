"""
Ashva Master Quantitative Research & Hypothesis Lab CLI
Unified institutional entry-point for Alpha discovery, multi-regime backtesting,
and statistical validation across Indian equity and derivative markets.

DEPRECATION NOTICE:
This script has been deprecated in favor of `scripts/research_alpha.py` to enforce a single canonical
research path for panel alphas. It now delegates all execution to `research_alpha.py`.
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 135)
    print("[!] DEPRECATION WARNING: scripts/run_hypothesis_lab.py is deprecated.")
    print("[!] Delegating to scripts/research_alpha.py to ensure canonical 77-stock panel validation.")
    print("=" * 135)
    
    # Forward the arguments as best effort
    cmd = [sys.executable, "scripts/research_alpha.py"]
    if "--all" in sys.argv:
        cmd.append("--all")
    else:
        # Try to find strategy arg
        try:
            strat_idx = sys.argv.index("--strategy")
            strat_val = sys.argv[strat_idx + 1]
            cmd.extend(["--alpha-id", strat_val])
        except (ValueError, IndexError):
            pass

    print(f"[*] Running: {' '.join(cmd)}\n")
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
