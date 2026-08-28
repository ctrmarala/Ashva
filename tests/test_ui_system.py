"""
Unit tests for Tab 4 System Observability in Ashva UI.
Verifies system overview, engine health, data pipeline indicators, trading indicators,
configuration exposure, strict secret redaction, provenance, log parsing, and stale diagnostics.
"""

import pytest
import pandas as pd
from src.ui.data_access import UIDataAccess


@pytest.fixture
def dal_instance():
    return UIDataAccess()


def test_system_health_overview(dal_instance):
    overview = dal_instance.get_system_health_overview()
    assert overview["overall_status"] in ["HEALTHY", "DEGRADED", "ERROR"]
    assert overview["version"] == "v1.0.0"
    assert "git_commit" in overview
    assert "components" in overview
    assert "DATA" in overview["components"]
    assert "ALPHA FACTORY" in overview["components"]
    assert "TRADING" in overview["components"]
    assert "REPLAY" in overview["components"]
    assert "PAPER" in overview["components"]
    assert "LIVE" in overview["components"]


def test_engine_health_metrics(dal_instance):
    engines = dal_instance.get_engine_health_metrics()
    assert len(engines) == 6
    engine_names = [e["engine"] for e in engines]
    assert "Data Ingestion Engine" in engine_names
    assert "Alpha Factory Research Engine" in engine_names
    assert "Trading Engine Core" in engine_names
    assert "Replay Execution Engine" in engine_names
    assert "Paper Trading Engine" in engine_names
    assert "Live Broker Execution Engine" in engine_names

    for e in engines:
        assert e["status"] in ["HEALTHY", "DEGRADED", "STANDBY", "OFFLINE / STANDBY", "ERROR"]
        assert "last_activity" in e
        assert "current_state" in e


def test_data_pipeline_health_indicators(dal_instance):
    indicators = dal_instance.get_data_pipeline_health_indicators()
    assert "duckdb_storage" in indicators
    assert "symbols_available" in indicators
    assert "data_freshness" in indicators
    assert "hygiene_audit" in indicators


def test_trading_engine_health_indicators(dal_instance):
    indicators = dal_instance.get_trading_engine_health_indicators()
    assert indicators["trading_engine_state"] == "STANDBY"
    assert indicators["active_alpha_contracts_count"] > 0
    assert "total_net_pnl" in indicators
    assert "current_equity" in indicators


def test_active_system_configuration_and_security(dal_instance):
    config = dal_instance.get_active_system_configuration()
    assert "fund_configuration" in config
    assert "market_hours" in config
    assert "risk_limits" in config
    assert "alpha_qualification_hurdles" in config
    assert "gateway_credentials_security_audit" in config

    # STRICT SECURITY AUDIT: Verify no raw secrets/passwords exist in the exposed dictionary
    config_str = str(config)
    assert "hKRDBcQF" not in config_str  # Ensure sample private keys never appear
    assert "eyJhbGciOi" not in config_str  # Ensure raw JWT tokens never appear
    
    audit = config["gateway_credentials_security_audit"]
    for key, val in audit.items():
        assert val in ["CONFIGURED (Protected)", "NOT CONFIGURED", "DuckDB + Apache Parquet (Local Columnar Store)"]


def test_system_version_provenance(dal_instance):
    prov = dal_instance.get_system_version_provenance()
    assert prov["ashva_version"] == "v1.0.0"
    assert "git_commit" in prov
    assert "git_branch" in prov
    assert "python_version" in prov
    assert "os_platform" in prov


def test_stale_and_broken_state_diagnostics(dal_instance):
    diag = dal_instance.get_stale_and_broken_state_diagnostics()
    assert "data_staleness_status" in diag
    assert "missing_config_files" in diag
    assert "database_integrity_issues" in diag
    assert "stale_wal_locks" in diag


def test_operational_logs_sanitization(dal_instance):
    df_logs = dal_instance.get_operational_logs_and_errors(limit=20)
    assert isinstance(df_logs, pd.DataFrame)
    assert not df_logs.empty
    assert "Timestamp" in df_logs.columns
    assert "Component" in df_logs.columns
    assert "Severity" in df_logs.columns
    assert "Message" in df_logs.columns

    # Verify that regex sanitization stripped any raw token/key
    for msg in df_logs["Message"]:
        assert "eyJhbGciOi" not in msg
        assert "Bearer ey" not in msg


def test_system_runtime_info(dal_instance):
    rt = dal_instance.get_system_runtime_info()
    assert "python_version" in rt
    assert "python_executable" in rt
    assert "os_platform" in rt
    assert "duckdb_database_path" in rt
    assert "process_id" in rt
