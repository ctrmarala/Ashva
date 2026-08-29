"""
Ashva Institutional Quant Performance Tearsheet & Report Generator
Generates comprehensive hedge-fund grade HTML/Markdown performance reports with risk ratios,
drawdown analysis, monthly return tables, and exact Indian regulatory tax deductions.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class QuantTearsheetGenerator:
    """
    Generates standalone institutional tearsheets for backtested and live strategies.
    """

    def __init__(self, output_dir: str = "data_lake/tearsheets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_tearsheet(
        self,
        result: Any,
        cpcv_sharpe: float = 0.0,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> str:
        """
        Builds full HTML Quant Tearsheet with dark theme styling.
        """
        total_gross = result.final_equity - result.initial_capital + result.total_taxes_paid
        trades_list = result.trade_list if hasattr(result, "trade_list") else []

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Ashva Quant Tearsheet - {result.strategy_id} ({result.symbol})</title>
    <style>
        :root {{
            --bg-base: #0a0e17;
            --bg-card: #131b2e;
            --border: #202d4a;
            --accent-green: #00d68f;
            --accent-red: #ff3d71;
            --accent-blue: #0095ff;
            --text-main: #f7f9fc;
            --text-muted: #8f9bb3;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        body {{ background-color: var(--bg-base); color: var(--text-main); padding: 32px; }}
        .header {{ border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 28px; }}
        .header h1 {{ font-size: 26px; font-weight: 800; color: var(--accent-blue); }}
        .header p {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 18px; }}
        .card h4 {{ font-size: 11px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }}
        .card .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
        .pos {{ color: var(--accent-green); }}
        .neg {{ color: var(--accent-red); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }}
        th {{ color: var(--text-muted); font-size: 11px; text-transform: uppercase; }}
        .section-title {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; color: var(--text-main); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ASHVA QUANTITATIVE TEARSHEET</h1>
        <p>Strategy: <strong>{result.strategy_id}</strong> | Target Asset: <strong>{result.symbol}</strong> | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")}</p>
    </div>

    <div class="grid">
        <div class="card">
            <h4>Net Cumulative ROI</h4>
            <div class="value {'pos' if result.net_roi_pct >= 0 else 'neg'}">{result.net_roi_pct:+.2f}%</div>
        </div>
        <div class="card">
            <h4>Net Profit (Post-Tax)</h4>
            <div class="value {'pos' if result.total_net_pnl >= 0 else 'neg'}">₹{result.total_net_pnl:+,.2f}</div>
        </div>
        <div class="card">
            <h4>Annualized Sharpe (IS)</h4>
            <div class="value">{result.sharpe_ratio:.2f}</div>
        </div>
        <div class="card">
            <h4>OOS CPCV Sharpe</h4>
            <div class="value">{cpcv_sharpe:.2f}</div>
        </div>
        <div class="card">
            <h4>Max Drawdown</h4>
            <div class="value neg">-{result.max_drawdown_pct:.2f}%</div>
        </div>
        <div class="card">
            <h4>Win Rate ({result.total_trades} trades)</h4>
            <div class="value">{result.win_rate_pct:.1f}%</div>
        </div>
        <div class="card">
            <h4>Net Profit Factor (IS)</h4>
            <div class="value">{result.net_profit_factor:.2f}</div>
        </div>
    </div>

    <div class="card" style="margin-bottom: 28px;">
        <div class="section-title">Institutional Tax & Cost Attribution (NSE Statutory Levies)</div>
        <table>
            <thead>
                <tr>
                    <th>Initial Capital</th>
                    <th>Final Equity</th>
                    <th>Gross Trading P&L</th>
                    <th>Angel Brokerage (₹20 cap)</th>
                    <th>STT (Securities Transaction Tax)</th>
                    <th>Total Statutory Taxes & Charges</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>₹{result.initial_capital:,.2f}</td>
                    <td>₹{result.final_equity:,.2f}</td>
                    <td>₹{total_gross:+,.2f}</td>
                    <td class="neg">₹{getattr(result, 'total_brokerage_paid', 0.0):,.2f}</td>
                    <td class="neg">₹{getattr(result, 'total_stt_paid', 0.0):,.2f}</td>
                    <td class="neg"><strong>₹{getattr(result, 'total_taxes_paid', 0.0):,.2f}</strong></td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <div class="section-title">Recent Trade Executions Ledger</div>
        <table>
            <thead>
                <tr>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Gross P&L</th>
                    <th>Net P&L (Post-Tax)</th>
                </tr>
            </thead>
            <tbody>
        """

        if not trades_list:
            html += "<tr><td colspan='8' style='text-align: center; color: var(--text-muted);'>No trades recorded in sample.</td></tr>"
        else:
            for tr in trades_list[-15:]:
                html += f"""
                    <tr>
                        <td>{str(tr.entry_time)[:16]}</td>
                        <td>{str(tr.exit_time)[:16]}</td>
                        <td><strong>{tr.side}</strong></td>
                        <td>{tr.quantity}</td>
                        <td>₹{tr.entry_price:.2f}</td>
                        <td>₹{tr.exit_price:.2f}</td>
                        <td class="{'pos' if tr.gross_pnl >= 0 else 'neg'}">₹{tr.gross_pnl:+,.2f}</td>
                        <td class="{'pos' if tr.net_pnl >= 0 else 'neg'}"><strong>₹{tr.net_pnl:+,.2f}</strong></td>
                    </tr>
                """


        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
        """

        filepath = self.output_dir / f"tearsheet_{result.strategy_id}_{result.symbol}.html"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return str(filepath)
