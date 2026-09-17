import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from scripts.run_alpha_period import evaluate_alpha_period

active_strats = [
    ('61_alpha', '15m'),
    ('62_alpha', '15m'),
    ('63_alpha', '15m'),
    ('65_alpha', '15m'),
    ('70_alpha', '5m'),
    ('74_alpha', '5m'),
]
total_net = 0.0
total_trades = 0
total_wins = 0

print("=" * 85)
print("ACTIVE ORTHOGONAL PORTFOLIO 30-DAY PERFORMANCE (NATIVE TIMEFRAMES)")
print("=" * 85)

for s, tf in active_strats:
    res = evaluate_alpha_period(s, timeframe=tf, start_date='2026-08-15', end_date='2026-09-15')
    total_net += res['net_pnl']
    total_trades += res['trades']
    total_wins += res['wins']
    t_cnt = res['trades']
    wr = res['win_rate_pct']
    net_p = res['net_pnl']
    pf = res['net_profit_factor']
    print(f"{s:<12} ({tf:<3}) | Trades: {t_cnt:2d} | Win Rate: {wr:5.1f}% | Net PnL: Rs {net_p:>+10.2f} | Net PF: {pf:4.2f}")

print("-" * 85)
roi = (total_net / 500000.0) * 100.0
tot_wr = (total_wins / max(1, total_trades)) * 100.0
print(f"TOTAL ACTIVE PORTFOLIO: Trades: {total_trades} | Win Rate: {tot_wr:.1f}% | Net PnL: Rs {total_net:+,.2f} | 30-Day Monthly ROI: {roi:+.2f}%")
print("=" * 85)
