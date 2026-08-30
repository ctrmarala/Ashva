"""
Ashva Dynamic Strategy Registry
Provides plug-and-play auto-discovery for all Alpha strategy classes in src/strategies/.
Any new alpha_*.py file placed in src/strategies/ is immediately accessible without manual registration.
Returns canonical 1-to-1 mapping of Strategy ID -> Strategy Class.
"""

import sys
import importlib
from pathlib import Path
from typing import Dict, Any, Type, Optional


_STRATEGY_CACHE: Optional[Dict[str, Type[Any]]] = None


def get_all_strategies(reload: bool = False) -> Dict[str, Type[Any]]:
    """
    Scans src/strategies/ for all alpha_*.py files, imports them dynamically,
    and returns a canonical mapping of StrategyID -> StrategyClass (1 entry per strategy).
    """
    global _STRATEGY_CACHE
    if _STRATEGY_CACHE is not None and not reload:
        return _STRATEGY_CACHE

    strat_dir = Path(__file__).parent
    current_file_stems = {
        p.stem for p in strat_dir.glob("*.py")
        if p.name not in ["base.py", "registry.py", "__init__.py"]
    }

    # Clean up sys.modules of any renamed/deleted strategy files
    for mod_key in list(sys.modules.keys()):
        if mod_key.startswith("src.strategies."):
            stem = mod_key.split(".")[-1]
            if stem not in current_file_stems and stem not in ["base", "registry", "__init__"]:
                del sys.modules[mod_key]

    strategies = {}

    for p in sorted(strat_dir.glob("*.py")):
        if p.name in ["base.py", "registry.py", "__init__.py"]:
            continue
        mod_name = f"src.strategies.{p.stem}"
        try:
            if mod_name in sys.modules and reload:
                mod = importlib.reload(sys.modules[mod_name])
            else:
                mod = importlib.import_module(mod_name)

            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (
                    isinstance(obj, type)
                    and hasattr(obj, "generate_signals")
                    and obj.__name__ not in ["BaseStrategy", "BaseHypothesis", "CrossSectionalHypothesis"]
                ):
                    strat_id = getattr(obj, "strategy_id", None) or attr
                    strategies[strat_id] = obj
                    break
        except Exception as e:
            print(f"[!] Warning: Could not auto-load strategy module {p.name}: {e}")

    _STRATEGY_CACHE = strategies
    return strategies


def get_strategy_by_name(name: str) -> Optional[Type[Any]]:
    """
    Retrieves a strategy class by its exact strategy_id, class name, or case-insensitive match.
    """
    strats = get_all_strategies()
    if name in strats:
        return strats[name]

    # Check class names or case-insensitive keys
    for k, cls in strats.items():
        if k.lower() == name.lower() or cls.__name__.lower() == name.lower():
            return cls

    return None
