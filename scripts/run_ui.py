"""
Ashva Observability UI Launcher
Starts the unified Streamlit observability dashboard on http://127.0.0.1:8501
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent
    dashboard_path = root_dir / "src" / "ui" / "dashboard.py"
    
    # Use the active python interpreter executable
    python_exe = sys.executable
    cmd = [python_exe, "-m", "streamlit", "run", str(dashboard_path), "--server.headless", "false"]
    
    print("=" * 70)
    print("?? ASHVA UNIFIED OBSERVABILITY UI")
    print(f"[*] Dashboard Target: {dashboard_path}")
    print("[*] Local URL:        http://localhost:8501")
    print("[*] Network URL:      http://127.0.0.1:8501")
    print("=" * 70)
    
    subprocess.run(cmd, cwd=str(root_dir))

if __name__ == "__main__":
    main()
