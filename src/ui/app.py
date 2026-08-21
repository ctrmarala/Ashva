"""
Ashva Real-Time Web Control Room & REST Telemetry Server
Institutional Dark-Theme Dashboard for monitoring live PnL, trade history ledger, risk gauges,
multi-mode telemetry (REPLAY, PAPER, LIVE), Trade Drill-Down ('Why did Ashva do this?'),
and the Alpha Cockpit comparing research metrics across lifecycle stages.
"""

from datetime import datetime
import json
import os
import sys
import sqlite3
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.trading.ledger import TradingLedger
from src.trading.live_rms import LiveRiskManager
from src.analytics.feedback import ResearchFeedbackAnalyzer

logger = None


class AshvaControlRoomHandler(SimpleHTTPRequestHandler):
    """
    Handles dashboard telemetry REST endpoints and UI page rendering.
    """

    ledger = TradingLedger()
    risk_manager = LiveRiskManager()
    feedback_analyzer = ResearchFeedbackAnalyzer(ledger=ledger)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            telemetry = self._get_telemetry_payload()
            self.wfile.write(json.dumps(telemetry).encode("utf-8"))
            return

        elif parsed.path == "/api/trade_drilldown":
            trade_id_str = params.get("id", ["1"])[0]
            trade_id = int(trade_id_str) if trade_id_str.isdigit() else 1
            drilldown = self.ledger.get_trade_drilldown(trade_id) or {}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(drilldown, default=str).encode("utf-8"))
            return

        elif parsed.path == "/api/alpha_cockpit":
            alpha_id = params.get("alpha_id", [None])[0]
            cockpit_data = self._get_alpha_cockpit_data(alpha_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(cockpit_data, default=str).encode("utf-8"))
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
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "KILL_SWITCH_ACTIVATED"}).encode("utf-8"))
            return

        elif parsed.path == "/api/reset_kill_switch":
            self.risk_manager.reset_kill_switch(reason="WEB_DASHBOARD_RESET_BUTTON")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "KILL_SWITCH_RESET"}).encode("utf-8"))
            return

        self.send_error(404)

    def _get_telemetry_payload(self) -> dict:
        """Fetches telemetry metrics from TradingLedger."""
        trades = self.ledger.get_trades(limit=50)
        today_realized_pnl = sum(t["net_pnl"] for t in trades)
        today_gross_pnl = sum(t["gross_pnl"] for t in trades)
        today_costs = sum(t["total_costs"] for t in trades)

        return {
            "timestamp": datetime.now().isoformat(),
            "mode": "REPLAY / PAPER / LIVE",
            "safety_state": self.risk_manager.safety_state.value,
            "kill_switch_active": self.risk_manager.kill_switch_active,
            "today_realized_pnl": round(today_realized_pnl, 2),
            "today_gross_pnl": round(today_gross_pnl, 2),
            "today_costs": round(today_costs, 2),
            "closed_trades_count": len(trades),
            "recent_trades": trades[:15],
        }

    def _get_alpha_cockpit_data(self, alpha_id: Optional[str]) -> dict:
        """Constructs multi-stage comparison for Alpha Cockpit."""
        trades = self.ledger.get_trades(alpha_id=alpha_id, limit=200)
        replay_trades = [t for t in trades if t.get("mode") == "REPLAY"]
        paper_trades = [t for t in trades if t.get("mode") == "PAPER"]
        live_trades = [t for t in trades if t.get("mode") == "LIVE"]

        def calc_stage_metrics(t_list):
            if not t_list:
                return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "pf": 0.0, "avg_trade": 0.0}
            wins = [t for t in t_list if t["net_pnl"] > 0]
            losses = [t for t in t_list if t["net_pnl"] < 0]
            gross_w = sum(t["gross_pnl"] for t in wins)
            gross_l = abs(sum(t["gross_pnl"] for t in losses))
            pf = (gross_w / gross_l) if gross_l > 0 else (99.0 if gross_w > 0 else 0.0)
            return {
                "trades": len(t_list),
                "win_rate": round(len(wins) / len(t_list) * 100.0, 1),
                "net_pnl": round(sum(t["net_pnl"] for t in t_list), 2),
                "pf": round(pf, 2),
                "avg_trade": round(sum(t["net_pnl"] for t in t_list) / len(t_list), 2),
            }

        return {
            "alpha_id": alpha_id or "ALL_ALPHAS",
            "stages": {
                "REPLAY": calc_stage_metrics(replay_trades),
                "PAPER": calc_stage_metrics(paper_trades),
                "LIVE": calc_stage_metrics(live_trades),
            },
            "research_hypotheses": self.feedback_analyzer.analyze_alpha_performance(alpha_id or "") if alpha_id else [],
        }

    def _render_dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Ashva Institutional Trading Engine Control Room</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #121824;
            --card-bg: #1a2234;
            --accent-blue: #00d2ff;
            --accent-green: #00e676;
            --accent-red: #ff1744;
            --accent-gold: #ffd600;
            --text-main: #e2e8f0;
            --text-dim: #94a3b8;
            --border-color: #2a364f;
        }
        body { font-family: 'Segoe UI', -apple-system, sans-serif; background: var(--bg-primary); color: var(--text-main); margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; }
        .title { font-size: 24px; font-weight: 700; color: var(--accent-blue); }
        .badge { padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; background: var(--card-bg); border: 1px solid var(--border-color); }
        .badge.active { color: var(--accent-green); border-color: var(--accent-green); }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
        .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 18px; }
        .card-title { font-size: 13px; color: var(--text-dim); text-transform: uppercase; margin-bottom: 8px; }
        .card-val { font-size: 26px; font-weight: 700; }
        .table-container { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { text-align: left; padding: 10px; color: var(--text-dim); border-bottom: 1px solid var(--border-color); }
        td { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .btn-kill { background: var(--accent-red); color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: 700; cursor: pointer; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">ASHVA QUANTITATIVE TRADING ENGINE</div>
            <div style="font-size: 12px; color: var(--text-dim);">Institutional Autonomous Execution Architecture</div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span class="badge active" id="mode-badge">UNIFIED TRADING ENGINE</span>
            <button class="btn-kill" onclick="triggerKill()">EMERGENCY KILL SWITCH</button>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Realized Net P&L</div>
            <div class="card-val" id="val-pnl" style="color: var(--accent-green);">₹0.00</div>
        </div>
        <div class="card">
            <div class="card-title">Safety State</div>
            <div class="card-val" id="val-safety" style="color: var(--accent-blue);">ACTIVE</div>
        </div>
        <div class="card">
            <div class="card-title">Closed Trades</div>
            <div class="card-val" id="val-trades">0</div>
        </div>
        <div class="card">
            <div class="card-title">Total Frictions / Taxes</div>
            <div class="card-val" id="val-costs" style="color: var(--accent-gold);">₹0.00</div>
        </div>
    </div>

    <div class="table-container">
        <h3 style="margin-top: 0; color: var(--accent-blue);">Authoritative Execution Ledger</h3>
        <table>
            <thead>
                <tr>
                    <th>Trade ID</th>
                    <th>Alpha ID</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Net P&L</th>
                    <th>MFE</th>
                    <th>MAE</th>
                    <th>Holding Bars</th>
                    <th>Mode</th>
                </tr>
            </thead>
            <tbody id="trades-body">
                <tr><td colspan="12" style="text-align: center; color: var(--text-dim);">Listening to Trading Ledger...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                document.getElementById('val-pnl').innerText = '₹' + data.today_realized_pnl.toLocaleString('en-IN', {minimumFractionDigits: 2});
                document.getElementById('val-pnl').style.color = data.today_realized_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
                document.getElementById('val-safety').innerText = data.safety_state;
                document.getElementById('val-trades').innerText = data.closed_trades_count;
                document.getElementById('val-costs').innerText = '₹' + data.today_costs.toLocaleString('en-IN', {minimumFractionDigits: 2});

                const tbody = document.getElementById('trades-body');
                if (data.recent_trades && data.recent_trades.length > 0) {
                    tbody.innerHTML = data.recent_trades.map(t => `
                        <tr>
                            <td>#${t.trade_id}</td>
                            <td><strong>${t.alpha_id}</strong></td>
                            <td>${t.symbol}</td>
                            <td><span style="color: ${t.side === 'BUY' ? 'var(--accent-green)' : 'var(--accent-red)'}">${t.side}</span></td>
                            <td>${t.quantity}</td>
                            <td>₹${t.entry_price.toFixed(2)}</td>
                            <td>₹${t.exit_price.toFixed(2)}</td>
                            <td style="font-weight: bold; color: ${t.net_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">₹${t.net_pnl.toFixed(2)}</td>
                            <td style="color: var(--accent-green);">₹${t.mfe.toFixed(2)}</td>
                            <td style="color: var(--accent-red);">₹${t.mae.toFixed(2)}</td>
                            <td>${t.holding_period_bars}</td>
                            <td><span class="badge">${t.mode}</span></td>
                        </tr>
                    `).join('');
                }
            } catch (err) {
                console.error('Failed to fetch telemetry:', err);
            }
        }
        async function triggerKill() {
            if (confirm('Are you sure you want to activate the Global Emergency Kill Switch?')) {
                await fetch('/api/kill_switch', {method: 'POST'});
                fetchTelemetry();
            }
        }
        setInterval(fetchTelemetry, 2000);
        fetchTelemetry();
    </script>
</body>
</html>
"""
