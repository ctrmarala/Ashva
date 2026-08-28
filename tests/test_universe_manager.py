"""
Unit tests for UniverseManager and universe-agnostic asset resolution in Ashva.
"""

from pathlib import Path
import pytest
from src.core.universe_manager import UniverseManager, get_universe_symbols, get_universe_name, get_benchmark_symbol
from src.ui.data_access import UIDataAccess


def test_universe_manager_default_resolution():
    um = UniverseManager(config_path="config/settings.yaml")
    symbols = um.get_universe_symbols()
    assert len(symbols) >= 50
    assert "RELIANCE" in symbols
    assert "INFY" in symbols
    assert "TCS" in symbols


def test_universe_manager_metadata():
    um = UniverseManager(config_path="config/settings.yaml")
    assert um.get_universe_name() in ["NIFTY 50", "NIFTY 75", "NIFTY 100"]
    assert um.get_benchmark_symbol() == "^NSEI"


def test_global_helper_functions():
    symbols = get_universe_symbols()
    assert len(symbols) >= 50
    assert get_universe_name() in ["NIFTY 50", "NIFTY 75", "NIFTY 100"]
    assert get_benchmark_symbol() == "^NSEI"


def test_dal_universe_integration():
    dal = UIDataAccess()
    assert dal.get_active_universe_name() in ["NIFTY 50", "NIFTY 75", "NIFTY 100"]
    assert dal.get_benchmark_symbol() == "^NSEI"
    overview = dal.get_data_overview()
    assert "universe_name" in overview
    assert overview["universe_name"] in ["NIFTY 50", "NIFTY 75", "NIFTY 100"]


def test_custom_universe_config_resolution(tmp_path):
    # Test that changing settings.yaml to a custom universe (e.g. NIFTY 100) resolves dynamically
    custom_yaml = tmp_path / "custom_settings.yaml"
    custom_yaml.write_text("""
universe:
  name: "NIFTY 100"
  benchmark: "^NSEI"
  symbols:
    - "INFY"
    - "TCS"
    - "RELIANCE"
    - "ZOMATO"
    - "JIOFIN"
""")
    um = UniverseManager(config_path=str(custom_yaml))
    assert um.get_universe_name() == "NIFTY 100"
    symbols = um.get_universe_symbols(duckdb_path=None)
    assert symbols == ["INFY", "JIOFIN", "RELIANCE", "TCS", "ZOMATO"]
