"""
Unit Tests for Write-Ahead Logging (WAL) State Machine
"""

import os
import shutil
import tempfile
import pytest
from src.core.state_machine import StateMachineWAL


@pytest.fixture
def temp_wal():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_wal.db")
    wal = StateMachineWAL(db_path=db_path)
    yield wal
    shutil.rmtree(temp_dir)


def test_portfolio_state_persistence(temp_wal):
    temp_wal.save_portfolio_state(
        cash=450000.0,
        equity=495000.0,
        daily_starting_equity=500000.0,
        peak_equity=505000.0,
        kill_switch_active=False,
    )

    loaded = temp_wal.load_portfolio_state()
    assert loaded is not None
    assert loaded["cash"] == 450000.0
    assert loaded["equity"] == 495000.0
    assert loaded["kill_switch_active"] is False


def test_open_positions_lifecycle(temp_wal):
    temp_wal.upsert_position(
        symbol="RELIANCE",
        side="LONG",
        quantity=50,
        entry_price=2500.0,
        entry_time="2026-01-01T10:00:00",
        strategy_id="ALPHA_ORB",
        stop_loss=2450.0,
    )

    positions = temp_wal.load_open_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "RELIANCE"
    assert positions[0]["quantity"] == 50

    # Remove position
    temp_wal.remove_position("RELIANCE")
    positions_after = temp_wal.load_open_positions()
    assert len(positions_after) == 0
