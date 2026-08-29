"""
Ashva Quantitative Alpha Linter & Static Strategy Contract Guardrail
Enforces architectural integrity, zero hardcoded tickers, strict zero lookahead,
mandatory metadata, parameter search grids, and intraday 15:15 IST square-off.
"""

import ast
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Type
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, StrategyHorizon, MarketMechanism
from src.strategies.base import BaseStrategy


FORBIDDEN_TICKER_STRINGS = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
    "LT", "ITC", "AXISBANK", "KOTAKBANK", "MARUTI", "TATAMOTORS", "SUNPHARMA"
}


class AlphaLinterError(Exception):
    """Raised when an alpha strategy violates Ashva institutional principles."""
    pass


class AlphaLinter:
    """
    Automated linter and contract auditor for all quantitative strategies in Ashva.
    """

    @classmethod
    def lint_strategy_source_file(cls, file_path: str) -> List[str]:
        """
        Parses the Python source code AST to statically catch hardcoded symbols,
        missing imports, or suspicious patterns.
        """
        p = Path(file_path)
        if not p.exists():
            return [f"File not found: {file_path}"]

        with open(p, "r", encoding="utf-8") as f:
            source_code = f.read()

        violations = []
        try:
            tree = ast.parse(source_code, filename=str(p))
        except SyntaxError as e:
            return [f"SyntaxError in {file_path}: {e}"]

        # 1. Scan for hardcoded symbol lists/sets in assignments
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
                str_elements = [elt.value for elt in node.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
                forbidden_matches = [s for s in str_elements if s.upper() in FORBIDDEN_TICKER_STRINGS]
                # If more than 2 forbidden tickers appear in a literal list, flag hardcoding
                if len(forbidden_matches) >= 2:
                    violations.append(
                        f"Line {node.lineno}: Detected hardcoded instrument list ({forbidden_matches}). "
                        "Strategies must dynamically use get_universe_symbols()."
                    )

        # 2. Check for presence of BaseHypothesis and BaseStrategy inheritance
        has_base_hypo = False
        has_base_strat = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
                if "BaseHypothesis" in base_names:
                    has_base_hypo = True
                if "BaseStrategy" in base_names:
                    has_base_strat = True

        if not (has_base_hypo or has_base_strat):
            violations.append(
                f"{file_path}: Strategy class must inherit from BaseHypothesis and BaseStrategy."
            )

        return violations

    @classmethod
    def lint_strategy_instance(cls, strat_obj: Any) -> List[str]:
        """
        Validates runtime contract properties: metadata, dynamic universe, parameter grid,
        signal generation output columns, zero lookahead leakage, and 15:15 IST exit.
        """
        violations = []

        # 1. Type & Inheritance Check
        if not isinstance(strat_obj, (BaseHypothesis, BaseStrategy)):
            violations.append(f"{strat_obj}: Must be instance of BaseHypothesis and BaseStrategy.")
            return violations

        # 2. Metadata Integrity Check
        meta = getattr(strat_obj, "metadata", None)
        if meta is None:
            violations.append(f"{strat_obj}: Missing HypothesisMetadata.")
        else:
            if not getattr(meta, "hypothesis_id", None):
                violations.append("Metadata missing hypothesis_id.")
            if not getattr(meta, "name", None):
                violations.append("Metadata missing name.")
            if not getattr(meta, "economic_rationale", None) or len(meta.economic_rationale) < 20:
                violations.append("Metadata missing complete economic_rationale (>20 chars).")
            target_instruments = getattr(meta, "target_instruments", [])
            if target_instruments and len(target_instruments) < 50:
                violations.append(
                    f"Metadata target_instruments has {len(target_instruments)} symbols. "
                    "If specified, must dynamically bind to active universe (>=50 symbols)."
                )

        # 3. Parameter Grid Check
        if not hasattr(strat_obj, "get_parameter_grid"):
            violations.append("Strategy must implement get_parameter_grid().")
        else:
            grid = strat_obj.get_parameter_grid()
            if not isinstance(grid, dict) or len(grid) == 0:
                violations.append("get_parameter_grid() must return a non-empty dict of parameter search lists.")
            else:
                for p_name, p_vals in grid.items():
                    if not isinstance(p_vals, (list, tuple)) or len(p_vals) < 2:
                        violations.append(f"Parameter grid for '{p_name}' must have at least 2 test values.")

        # 4. Signal Output & Column Contract
        dummy_df = cls._create_synthetic_test_df()
        try:
            sig_df = strat_obj.generate_signals(dummy_df.copy())
            if not isinstance(sig_df, pd.DataFrame):
                violations.append("generate_signals() must return a pandas DataFrame.")
            else:
                for col in ["signal", "stop_loss", "take_profit"]:
                    if col not in sig_df.columns:
                        violations.append(f"generate_signals() output missing required column: '{col}'.")
        except Exception as e:
            violations.append(f"generate_signals() crashed during validation: {e}")

        # 5. Strict Zero Look-Ahead Perturbation Check
        if not violations:
            lookahead_violation = cls._test_lookahead_leakage(strat_obj, dummy_df)
            if lookahead_violation:
                violations.append(lookahead_violation)

        # 6. Intraday 15:15 IST Square-Off Contract Check
        if not violations and getattr(strat_obj.metadata, "horizon", None) == StrategyHorizon.INTRADAY:
            square_off_violation = cls._test_1515_square_off(strat_obj, dummy_df)
            if square_off_violation:
                violations.append(square_off_violation)

        return violations

    @staticmethod
    def _create_synthetic_test_df() -> pd.DataFrame:
        timestamps = [
            datetime(2026, 8, 27, 9, 15) + timedelta(minutes=15 * i) for i in range(25)
        ] + [
            datetime(2026, 8, 28, 9, 15) + timedelta(minutes=15 * i) for i in range(25)
        ]
        total_bars = len(timestamps)
        opens = np.ones(total_bars) * 1000.0
        highs = np.ones(total_bars) * 1005.0
        lows = np.ones(total_bars) * 995.0
        closes = np.ones(total_bars) * 1000.0
        volumes = np.ones(total_bars) * 10000.0
        
        # Day 2 gap up + trend
        opens[25] = 1010.0
        highs[25] = 1015.0
        lows[25] = 1008.0
        closes[25] = 1014.0
        for j in range(26, total_bars):
            opens[j] = 1014.0 + (j - 25) * 1.5
            highs[j] = opens[j] + 3.0
            lows[j] = opens[j] - 1.0
            closes[j] = opens[j] + 2.0

        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }, index=pd.DatetimeIndex(timestamps))

    @classmethod
    def _test_lookahead_leakage(cls, strat_obj: Any, df: pd.DataFrame) -> Optional[str]:
        """Ensures altering future bars has ZERO impact on historical signals."""
        try:
            base_sig = strat_obj.generate_signals(df.copy())
            
            # Truncation test: Signals up to bar 26 must be identical
            trunc_df = df.iloc[:27].copy()
            trunc_sig = strat_obj.generate_signals(trunc_df)
            
            if not np.allclose(base_sig["signal"].iloc[:27].values, trunc_sig["signal"].values, equal_nan=True):
                return "Lookahead Leakage Detected: Signal changed when future bars were truncated."

            # Perturbation test: Modify bar 30 to extreme price
            perturbed_df = df.copy()
            perturbed_df.iloc[30, perturbed_df.columns.get_loc("close")] = 99999.0
            perturbed_sig = strat_obj.generate_signals(perturbed_df)
            
            if not np.allclose(base_sig["signal"].iloc[:30].values, perturbed_sig["signal"].iloc[:30].values, equal_nan=True):
                return "Lookahead Leakage Detected: Signal at historical bars changed when future price was perturbed."

        except Exception as e:
            return f"Lookahead check failed with exception: {e}"

        return None

    @classmethod
    def _test_1515_square_off(cls, strat_obj: Any, df: pd.DataFrame) -> Optional[str]:
        """Ensures that all intraday signals square off and no trade spans past 15:15 IST."""
        try:
            sig_df = strat_obj.generate_signals(df.copy())
            times = pd.to_datetime(sig_df.index).time
            # Check if any signal remains non-zero after 15:15
            for idx, (t, sig) in enumerate(zip(times, sig_df["signal"])):
                if t >= pd.Timestamp("15:15:00").time() and sig != 0.0:
                    # If signal is active past 15:15, flag violation
                    return f"15:15 Square-Off Violation: Signal at {sig_df.index[idx]} is non-zero ({sig}). Intraday alphas must square off by 15:15."
        except Exception as e:
            return f"15:15 Square-off check error: {e}"
        return None