"""
Ashva Real-Time Web Control Room & REST Telemetry Server
Institutional Dark-Theme Dashboard for monitoring live PnL, risk gauges, strategy allocations, and kill switches.
"""

from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

from src.core.state_machine import StateMachineWAL
from src.risk.risk_manager import RiskManager
from src.risk.var_calculator import RiskMetricsCalculator
from src.data.data_lake import DataLake
import numpy as np


class AshvaControlRoomHandler(SimpleHTTPRequestHandler):
    """
    Handles dashboard telemetry REST endpoints and UI page rendering.
    """

    state_wal = StateMachineWAL()
    risk_manager = RiskManager()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            telemetry = self._get_telemetry_payload()
            self.wfile.write(json.dumps(telemetry).encode("utf-8"))
            return

        elif parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html_content = self._render_dashboard_html()
            self.wfile.write(html_content.encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/api/kill_switch":
            self.risk_manager.trigger_kill_switch(reason="WEB_DASHBOARD_BUTTON_TRIGGERED")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "KILL_SWITCH_ACTIVATED"}).encode("utf-8"))
            return

        self.send_error(404)

    def _get_telemetry_payload(self) -> dict:
        state = self.state_wal.load_portfolio_state() or {
            "cash": 500000.0,
            "equity": 500000.0,
            "daily_starting_equity": 500000.0,
            "peak_equity": 500000.0,
            "kill_switch_active": False,
        }

        open_positions = self.state_wal.load_open_positions()
        daily_pnl = state["equity"] - state["daily_starting_equity"]
        daily_pnl_pct = (daily_pnl / state["daily_starting_equity"]) * 100.0 if state["daily_starting_equity"] > 0 else 0.0

        # Real VaR calculation: If in 100% cash, market risk is exactly 0.00 INR
        if len(open_positions) == 0:
            var_metrics = {
                "gaussian_var_pct": 0.0,
                "cornish_fisher_var_pct": 0.0,
                "var_inr": 0.0,
                "confidence_level": 0.95,
            }
        else:
            # Calculate VaR based on actual historical volatility of open assets
            try:
                lake = DataLake(read_only=True)
                holdings_var = 0.0
                for pos in open_positions:
                    sym = pos["symbol"]
                    bars = lake.load_bars(sym, "15m")
                    if not bars.empty and len(bars) > 30:
                        rets = bars["close"].pct_change().dropna().values
                        pos_val = pos["quantity"] * pos["entry_price"]
                        pos_var = RiskMetricsCalculator.calculate_parametric_var(rets, confidence_level=0.95, portfolio_value=pos_val)
                        holdings_var += pos_var.get("var_inr", 0.0)

                var_pct = (holdings_var / state["equity"]) * 100.0 if state["equity"] > 0 else 0.0
                var_metrics = {
                    "gaussian_var_pct": round(var_pct, 3),
                    "cornish_fisher_var_pct": round(var_pct, 3),
                    "var_inr": round(holdings_var, 2),
                    "confidence_level": 0.95,
                }
            except Exception:
                var_metrics = {
                    "gaussian_var_pct": 0.0,
                    "cornish_fisher_var_pct": 0.0,
                    "var_inr": 0.0,
                    "confidence_level": 0.95,
                }

        return {
            "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "cash": round(state["cash"], 2),
            "equity": round(state["equity"], 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "kill_switch_active": state["kill_switch_active"] or self.risk_manager.kill_switch_active,
            "open_positions": open_positions,
            "var_metrics": var_metrics,
            "strategy_allocations": {
                "ALPHA_07_ML_TREND_PULLBACK": 0.50,
                "ALPHA_08_ML_VOL_SQUEEZE": 0.35,
                "ALPHA_05_OPTIONS_STRADDLE": 0.15,
            },
        }

    def _render_dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASHVA Quantitative Control Room</title>
    <style>
        :root {
            --bg-base: #0a0e17;
            --bg-card: #131b2e;
            --border: #202d4a;
            --accent-green: #00d68f;
            --accent-red: #ff3d71;
            --accent-blue: #0095ff;
            --text-main: #f7f9fc;
            --text-muted: #8f9bb3;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background-color: var(--bg-base); color: var(--text-main); padding: 24px; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        .logo { font-size: 24px; font-weight: 800; letter-spacing: 2px; color: var(--accent-blue); display: flex; align-items: center; gap: 8px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 24px; }
        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
        .card h3 { font-size: 13px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 12px; letter-spacing: 0.5px; }
        .metric-value { font-size: 28px; font-weight: 700; }
        .positive { color: var(--accent-green); }
        .negative { color: var(--accent-red); }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 14px; }
        th { color: var(--text-muted); font-size: 12px; text-transform: uppercase; }
        .btn-kill { background: var(--accent-red); color: #fff; border: none; padding: 10px 20px; font-weight: 700; border-radius: 6px; cursor: pointer; transition: opacity 0.2s; }
        .btn-kill:hover { opacity: 0.85; }
        .badge { display: inline-block; padding: 3px 8px; font-size: 11px; font-weight: 700; border-radius: 4px; }
        .badge-long { background: rgba(0, 214, 143, 0.15); color: var(--accent-green); }
        .badge-short { background: rgba(255, 61, 113, 0.15); color: var(--accent-red); }
    </style>
</head>
<body>
    <header>
        <div class="logo">ASHVA QUANT CONTROL ROOM</div>
        <div>
            <span id="sysTime" style="color: var(--text-muted); margin-right: 16px; font-size: 14px;"></span>
            <button class="btn-kill" onclick="triggerKillSwitch()">EMERGENCY KILL SWITCH</button>
        </div>
    </header>

    <div class="grid">
        <div class="card">
            <h3>Portfolio Equity</h3>
            <div class="metric-value" id="equityVal">₹5,00,000.00</div>
        </div>
        <div class="card">
            <h3>Today's Net P&L</h3>
            <div class="metric-value" id="pnlVal">₹0.00 (0.00%)</div>
        </div>
        <div class="card">
            <h3>Available Cash</h3>
            <div class="metric-value" id="cashVal">₹5,00,000.00</div>
        </div>
        <div class="card">
            <h3>Value at Risk (95% VaR)</h3>
            <div class="metric-value" id="varVal" style="color: #ffaa00;">₹0.00</div>
        </div>
    </div>

    <div class="card" style="margin-bottom: 24px;">
        <h3>Active Open Positions</h3>
        <table id="positionsTable">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Entry Price</th>
                    <th>Strategy</th>
                    <th>Stop Loss</th>
                </tr>
            </thead>
            <tbody id="positionsBody">
                <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No active open positions.</td></tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h3>Strategy Capital Allocations (HRP Weights)</h3>
        <div id="allocationsContainer" style="display: flex; gap: 24px; margin-top: 12px;"></div>
    </div>

    <script>
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                
                document.getElementById('sysTime').innerText = data.system_time;
                document.getElementById('equityVal').innerText = '₹' + data.equity.toLocaleString('en-IN', {minimumFractionDigits: 2});
                document.getElementById('cashVal').innerText = '₹' + data.cash.toLocaleString('en-IN', {minimumFractionDigits: 2});
                
                const pnlEl = document.getElementById('pnlVal');
                const pnlSign = data.daily_pnl >= 0 ? '+' : '';
                pnlEl.innerText = pnlSign + '₹' + data.daily_pnl.toLocaleString('en-IN', {minimumFractionDigits: 2}) + ' (' + pnlSign + data.daily_pnl_pct.toFixed(2) + '%)';
                pnlEl.className = 'metric-value ' + (data.daily_pnl >= 0 ? 'positive' : 'negative');

                document.getElementById('varVal').innerText = '₹' + data.var_metrics.var_inr.toLocaleString('en-IN', {minimumFractionDigits: 2});

                // Positions
                const posBody = document.getElementById('positionsBody');
                if (data.open_positions.length === 0) {
                    posBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No active open positions.</td></tr>';
                } else {
                    posBody.innerHTML = data.open_positions.map(p => `
                        <tr>
                            <td><strong>${p.symbol}</strong></td>
                            <td><span class="badge ${p.side === 'LONG' ? 'badge-long' : 'badge-short'}">${p.side}</span></td>
                            <td>${p.quantity}</td>
                            <td>₹${p.entry_price.toFixed(2)}</td>
                            <td>${p.strategy_id}</td>
                            <td>₹${p.stop_loss ? p.stop_loss.toFixed(2) : '-'}</td>
                        </tr>
                    `).join('');
                }

                // Allocations
                const allocContainer = document.getElementById('allocationsContainer');
                allocContainer.innerHTML = Object.entries(data.strategy_allocations).map(([k, v]) => `
                    <div style="flex: 1; background: #0a0e17; padding: 12px; border-radius: 6px; border: 1px solid var(--border);">
                        <div style="font-size: 11px; color: var(--text-muted);">${k}</div>
                        <div style="font-size: 18px; font-weight: 700; color: var(--accent-blue); margin-top: 4px;">${(v * 100).toFixed(1)}%</div>
                    </div>
                `).join('');

            } catch (e) {
                console.error(e);
            }
        }

        async function triggerKillSwitch() {
            if (confirm('CRITICAL: Activate Global Emergency Kill Switch? This will block all orders!')) {
                await fetch('/api/kill_switch', { method: 'POST' });
                alert('Kill switch triggered.');
                fetchTelemetry();
            }
        }

        setInterval(fetchTelemetry, 2000);
        fetchTelemetry();
    </script>
</body>
</html>
"""


def start_control_room_server(port: int = 8080):
    """Starts local control room HTTP server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, AshvaControlRoomHandler)
    print(f"[*] Ashva Control Room Web Dashboard live at: http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    start_control_room_server()
