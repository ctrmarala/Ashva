"""
Ashva Dynamic Strategy Registry
Provides plug-and-play auto-discovery for all Alpha strategy classes in src/strategies/.
Any new alpha_*.py file placed in src/strategies/ is immediately accessible without manual registration.
"""

import importlib
from pathlib import Path
from typing import Dict, Any, Type, Optional


_STRATEGY_CACHE: Optional[Dict[str, Type[Any]]] = None


def get_all_strategies(reload: bool = False) -> Dict[str, Type[Any]]:
    """
    Scans src/strategies/ for all alpha_*.py files, imports them dynamically,
    and returns a mapping of StrategyName -> StrategyClass.
    """
    global _STRATEGY_CACHE
    if _STRATEGY_CACHE is not None and not reload:
        return _STRATEGY_CACHE

    strategies = {}
    strat_dir = Path(__file__).parent

    for p in sorted(strat_dir.glob("alpha_*.py")):
        mod_name = f"src.strategies.{p.stem}"
        try:
            mod = importlib.import_module(mod_name)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (
                    isinstance(obj, type)
                    and attr.startswith("Alpha")
                    and hasattr(obj, "generate_signals")
                ):
                    strategies[attr] = obj
                    break
        except Exception as e:
            print(f"[!] Warning: Could not auto-load strategy module {p.name}: {e}")

    _STRATEGY_CACHE = strategies
    return strategies


def get_strategy_by_name(name: str) -> Optional[Type[Any]]:
    """Retrieves a strategy class by its exact class name or fuzzy match."""
    strats = get_all_strategies()
    if name in strats:
        return strats[name]

    normalized_map = {k.lower(): v for k, v in strats.items()}
    return normalized_map.get(name.lower())
