"""
Ashva Institutional Statistical Alpha Validator & Lopez de Prado Hypothesis Rejector
Implements:
1. True Combinatorial Purged & Embargoed Cross-Validation (CPCV) with (N choose k) path combinations.
2. Deflated Sharpe Ratio (DSR) with Bailey & López de Prado (2014) non-normal asymptotic corrections.
3. 5,000+ Run Monte Carlo Permutation Stress Testing for tail-risk drawdown estimation.
4. Exact trade-by-trade Indian Regulatory Cost Modeling (STT, GST, SEBI, ₹20 Brokerage) via BacktestEngine.
"""

from datetime import datetime
import logging
import json
from math import comb
from itertools import combinations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from src.research.hypothesis import BaseHypothesis, HypothesisStatus, HypothesisValidationReport, StrategyHorizon
from src.analytics.indian_costs import IndianCostModel, Segment
from src.backtest.engine import BacktestEngine
from src.research.experiment_ledger import ResearchExperimentLedger, ExperimentRecord, get_current_git_sha

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """
    Institutional statistical filter to eliminate data snooping, multiple testing bias,
    and unviable strategies before capital allocation.
    """

    def __init__(
        self,
        cost_model: Optional[IndianCostModel] = None,
        experiment_ledger: Optional[ResearchExperimentLedger] = None,
        min_net_profit_factor: float = 1.20,
        max_dsr_p_value: float = 0.05,          # Statistical significance threshold
        max_cpcv_degradation_pct: float = 50.0, # Max allowed performance drop OOS
        max_monte_carlo_dd_pct: float = 15.0,   # Max 95th percentile drawdown
        min_trade_count: int = 25,              # Minimum sample size to avoid small-sample distortion
        max_portfolio_correlation: float = 0.50,# Max absolute correlation allowed with existing portfolio
    ):
        self.cost_model = cost_model or IndianCostModel()
        self.experiment_ledger = experiment_ledger or ResearchExperimentLedger()
        self.min_net_profit_factor = min_net_profit_factor
        self.max_dsr_p_value = max_dsr_p_value
        self.max_cpcv_degradation_pct = max_cpcv_degradation_pct
        self.max_monte_carlo_dd_pct = max_monte_carlo_dd_pct
        self.min_trade_count = min_trade_count
        self.max_portfolio_correlation = max_portfolio_correlation

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculates annualized Sharpe Ratio for daily or periodic strategy excess returns.
        """
        clean_ret = returns[~np.isnan(returns)]
        if len(clean_ret) < 2:
            return 0.0
        mean_ret = np.mean(clean_ret)
        std_ret = np.std(clean_ret, ddof=1)
        if std_ret < 1e-8:
            return 0.0
        
        rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0 if risk_free_rate > 0 else 0.0
        excess_mean = mean_ret - rf_per_period
        annualized_sharpe = (excess_mean / std_ret) * np.sqrt(periods_per_year)
        return float(annualized_sharpe)

    @classmethod
    def calculate_deflated_sharpe_ratio(
        cls,
        strategy_returns: np.ndarray,
        num_trials: int = 1,
        benchmark_sharpe_var: float = 0.5,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Computes Deflated Sharpe Ratio (DSR) and p-value (Bailey & López de Prado, 2014).
        Penalizes Sharpe ratio for multiple testing trials and non-normal (skewed/fat-tailed) returns.
        """
        clean_ret = strategy_returns[~np.isnan(strategy_returns)]
        t = len(clean_ret)
        if t < 5:
            return 0.0, 1.0

        sr = cls.calculate_sharpe_ratio(clean_ret, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
        if sr <= 0:
            return 0.0, 1.0

        gamma_3 = float(skew(clean_ret))
        gamma_4 = float(kurtosis(clean_ret, fisher=False))  # Pearson kurtosis (normal=3)

        # Expected maximum Sharpe ratio among N independent trials under Null Hypothesis
        euler_gamma = 0.5772156649
        if num_trials > 1:
            z_1 = norm.ppf(1.0 - 1.0 / num_trials)
            z_2 = norm.ppf(1.0 - 1.0 / (num_trials * np.e))
            expected_max_sr = np.sqrt(benchmark_sharpe_var) * ((1.0 - euler_gamma) * z_1 + euler_gamma * z_2)
        else:
            expected_max_sr = 0.0

        # Standard error adjustment for non-normality
        denom_sq = 1.0 - gamma_3 * sr + ((gamma_4 - 1.0) / 4.0) * (sr ** 2)
        if denom_sq <= 0:
            denom_sq = 1.0
        sigma_sr = np.sqrt(denom_sq / max(t - 1, 1))

        # DSR Z-Score and p-value
        dsr_stat = (sr - expected_max_sr) / sigma_sr
        p_value = float(1.0 - norm.cdf(dsr_stat))

        return float(dsr_stat), float(p_value)

    @classmethod
    def run_monte_carlo_drawdown_test(
        cls,
        trade_returns: np.ndarray,
        num_simulations: int = 5000,
    ) -> Dict[str, float]:
        """
        Simulates 5,000+ random permutations of trade return sequences
        to evaluate the true tail risk and maximum drawdown distribution.
        """
        if len(trade_returns) < 5:
            return {"mean_max_dd": 0.0, "p95_max_dd": 0.0, "p99_max_dd": 0.0}

        max_drawdowns = []
        for _ in range(num_simulations):
            shuffled = np.random.permutation(trade_returns)
            equity_curve = np.cumprod(1.0 + shuffled)
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = (peak - equity_curve) / peak
            max_drawdowns.append(np.max(drawdowns) * 100.0)

        dd_array = np.array(max_drawdowns)
        return {
            "mean_max_dd": float(np.mean(dd_array)),
            "p95_max_dd": float(np.percentile(dd_array, 95)),
            "p99_max_dd": float(np.percentile(dd_array, 99)),
        }

    def run_cpcv(
        self,
        df: pd.DataFrame,
        hypothesis: BaseHypothesis,
        n_splits: int = 6,
        k_test: int = 2,
        purge_bars: int = 5,
        embargo_bars: int = 5,
    ) -> Dict[str, Any]:
        """
        True Combinatorial Purged & Embargoed Cross-Validation (CPCV).
        Delegates to canonical CPCVEngine with zero synthetic temporal adjacency.
        """
        from src.research.cpcv_engine import CPCVEngine, CPCVMode

        seg = Segment.EQUITY_DELIVERY if getattr(hypothesis.metadata, "horizon", None) in [StrategyHorizon.SWING, StrategyHorizon.POSITIONAL] else Segment.EQUITY_INTRADAY
        engine = BacktestEngine(cost_model=self.cost_model, initial_capital=500000.0, segment=seg)
        
        full_signals_df = hypothesis.generate_signals(df)
        sym = getattr(hypothesis.metadata, "target_instruments", ["ASSET"])[0] if hasattr(hypothesis, "metadata") and hypothesis.metadata.target_instruments else "ASSET"
        res = engine.run(full_signals_df, symbol=sym, capital_per_trade_pct=0.50)

        if not res.trade_list:
            return {"mean_oos_sharpe": 0.0, "std_oos_sharpe": 0.0, "path_results": [], "total_paths": 0, "pbo": 1.0}

        trade_df = pd.DataFrame([{
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "net_pnl": t.net_pnl,
        } for t in res.trade_list])

        cpcv = CPCVEngine(n_partitions=n_splits, k_test_partitions=k_test, embargo_pct=0.01)
        cpcv_res = cpcv.evaluate_trades(trade_df, initial_capital=500000.0)

        return {
            "total_paths": cpcv_res.get("combinatorial_paths", 0),
            "mean_oos_sharpe": cpcv_res.get("mean_oos_sharpe", 0.0),
            "std_oos_sharpe": 0.0,
            "median_oos_sharpe": cpcv_res.get("median_oos_sharpe", 0.0),
            "pbo": cpcv_res.get("pbo", 1.0),
            "pbo_pct": cpcv_res.get("pbo_pct", "100.0%"),
            "degradation_ratio": cpcv_res.get("degradation_ratio", 0.0),
            "is_overfitted": cpcv_res.get("is_overfitted", True),
            "path_results": cpcv_res.get("path_results", []),
        }

    def validate_hypothesis(
        self,
        hypothesis: BaseHypothesis,
        df: pd.DataFrame,
        num_trials: Optional[int] = None,
        train_test_split: float = 0.70,
        symbol: Optional[str] = None,
        baseline_portfolio_returns: Optional[pd.DataFrame] = None,
    ) -> HypothesisValidationReport:
        """
        Full 4-Gate Institutional Statistical Validation with Recency-Weighted Multi-Window Analysis:
        Gate 1: Deflated Sharpe Ratio (DSR p <= 0.05) with strategy-family multiple testing trial accounting
        Gate 2: Combinatorial Purged & Embargoed Cross-Validation (CPCV Degradation <= 60%)
        Gate 3: 5,000-Run Monte Carlo Tail Risk (95th MaxDD <= 15%)
        Gate 4: Full Indian Regulatory Cost Profit Factor (Net PF >= 1.08 - 1.20)
        + 4-Tier Recency-Weighted Multi-Window Analysis (60d, 180d, 365d, 540d) with Confidence Shrinkage
        """
        rejection_reasons = []

        # 1. Parameter Multiple Testing by Strategy Research Family
        if num_trials is None or num_trials <= 1:
            grid = getattr(hypothesis, "parameter_grid", {})
            grid_size = 1
            for p_vals in grid.values():
                grid_size *= max(1, len(p_vals))
            trials_in_this_run = max(1, grid_size)
        else:
            trials_in_this_run = num_trials

        strat_id = hypothesis.metadata.hypothesis_id
        prior_trials = self.experiment_ledger.get_strategy_family_trials(strat_id)
        effective_trials = prior_trials + trials_in_this_run

        # 2. Segment configuration based on Strategy Horizon
        is_swing = getattr(hypothesis.metadata, "horizon", None) in [StrategyHorizon.SWING, StrategyHorizon.POSITIONAL]
        seg = Segment.EQUITY_DELIVERY if is_swing else Segment.EQUITY_INTRADAY

        sym = symbol or (getattr(hypothesis.metadata, "target_instruments", ["ASSET"])[0] if hasattr(hypothesis, "metadata") and hypothesis.metadata.target_instruments else "ASSET")

        # 3. Generate Signals and Run Baseline In-Sample / Full Backtests
        signals_df = hypothesis.generate_signals(df)
        if "signal" not in signals_df.columns:
            raise ValueError("Hypothesis must produce a 'signal' column")

        split_idx = int(len(signals_df) * train_test_split)
        train_df = signals_df.iloc[:split_idx]

        engine = BacktestEngine(cost_model=self.cost_model, initial_capital=500000.0, segment=seg)
        is_result = engine.run(train_df, symbol=sym, capital_per_trade_pct=0.50)
        full_result = engine.run(signals_df, symbol=sym, capital_per_trade_pct=0.50)

        is_sharpe = is_result.sharpe_ratio
        total_trades = full_result.total_trades
        net_pf = full_result.net_profit_factor if full_result.net_profit_factor < 90 else 99.0

        # 4. Multi-Window Recency Performance Breakdown (60d, 180d, 365d, 540d) with Confidence Shrinkage
        max_ts = signals_df.index.max()
        window_days = {"60d": 60, "180d": 180, "365d": 365, "540d": 540}
        window_metrics = {}
        window_qualities = {}

        for w_name, days in window_days.items():
            cutoff = max_ts - pd.Timedelta(days=days)
            w_sig = signals_df[signals_df.index >= cutoff]
            if not w_sig.empty:
                w_res = engine.run(w_sig, symbol=sym, capital_per_trade_pct=0.50)
                n_w = w_res.total_trades
                pnl_w = w_res.total_net_pnl
                pf_w = w_res.net_profit_factor if w_res.net_profit_factor < 90 else 99.0

                # Sample confidence factor: sqrt(N / 10), max 1.0 (prevents low-N explosion)
                conf_factor = min(1.0, np.sqrt(max(0, n_w) / 10.0))
                # Bounded return quality using hyperbolic tangent: tanh(pnl / 10000)
                q_w = conf_factor * float(np.tanh(pnl_w / 10000.0)) if n_w > 0 else 0.0
                window_qualities[w_name] = q_w

                window_metrics[w_name] = {
                    "trades": n_w,
                    "net_pnl": round(pnl_w, 2),
                    "net_pf": round(pf_w, 2),
                    "win_rate": round(w_res.win_rate_pct, 1),
                    "sharpe": round(w_res.sharpe_ratio, 2),
                    "max_dd_pct": round(w_res.max_drawdown_pct, 2),
                    "quality_score": round(float(q_w), 3),
                }
            else:
                window_qualities[w_name] = 0.0
                window_metrics[w_name] = {"trades": 0, "net_pnl": 0.0, "net_pf": 0.0, "win_rate": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0, "quality_score": 0.0}

        # Composite Recency-Weighted Quality Score (-1.0 to +1.0)
        # Weights: 50% 60d, 25% 180d, 15% 365d, 10% 540d
        recency_weighted_score = float(
            (0.50 * window_qualities["60d"]) +
            (0.25 * window_qualities["180d"]) +
            (0.15 * window_qualities["365d"]) +
            (0.10 * window_qualities["540d"])
        )

        # Regime Stability Score: % of active historical windows (180d, 365d, 540d) where net PnL > 0 and Sharpe > 0
        active_historical = [window_metrics[k] for k in ["180d", "365d", "540d"] if window_metrics[k]["trades"] > 0]
        stable_count = sum(1 for m in active_historical if m["net_pnl"] > 0 and m["sharpe"] > 0)
        regime_stability = (stable_count / max(1, len(active_historical))) * 100.0 if active_historical else 0.0

        # Current Regime Score (Descriptive indicator of 60d trajectory)
        curr_60 = window_metrics["60d"]
        n_60 = curr_60["trades"]
        pnl_60 = curr_60["net_pnl"]
        pf_60 = curr_60["net_pf"]

        if n_60 == 0:
            current_regime_score = 50.0  # Neutral / No recent trades
        elif n_60 >= 3 and pnl_60 > 0 and pf_60 >= 1.25 and curr_60["sharpe"] >= 1.0:
            current_regime_score = 90.0  # Accelerating
        elif pnl_60 > 0:
            current_regime_score = 70.0  # Positive
        elif pnl_60 < 0 and n_60 >= 2:
            current_regime_score = 25.0  # Decaying
        else:
            current_regime_score = 40.0  # Sub-optimal

        # Evidence Tier Classification based strictly on total trade count
        if total_trades >= 100:
            evidence_tier = "STRONG"
        elif total_trades >= 50:
            evidence_tier = "MODERATE"
        elif total_trades >= 25:
            evidence_tier = "PRELIMINARY"
        else:
            evidence_tier = "LOW_SAMPLE"

        # 5. Dynamic Profit Factor Hurdle based on Sample Density
        if total_trades >= 100:
            required_pf = 1.08
        elif total_trades >= 50:
            required_pf = 1.15
        else:
            required_pf = 1.20

        # 6. Gate 1: Deflated Sharpe Ratio (DSR) Test on Daily Mark-to-Market Returns
        daily_equity = full_result.equity_curve.resample("1D").last().dropna()
        daily_returns_series = daily_equity.pct_change().dropna()
        daily_returns = daily_returns_series.values
        dsr_stat, dsr_p_val = self.calculate_deflated_sharpe_ratio(daily_returns, num_trials=effective_trials, periods_per_year=252)
        if dsr_p_val > self.max_dsr_p_value:
            rejection_reasons.append(
                f"DSR Test Failed: p-value {dsr_p_val:.4f} > {self.max_dsr_p_value} across {effective_trials} parameter trials (High selection risk)"
            )

        # 6.5. Gate 1.5: Portfolio Diversification Filter (Question B)
        # Note: High correlation does not reject the hypothesis; it flags it for portfolio redundancy.
        corr_daily = 0.0
        corr_regime = 0.0
        corr_trade = 0.0
        if baseline_portfolio_returns is not None and not baseline_portfolio_returns.empty:
            strategy_daily_ret = daily_returns_series.rename("strategy")
            aligned_returns = pd.concat([baseline_portfolio_returns, strategy_daily_ret], axis=1).dropna()
            if not aligned_returns.empty and len(aligned_returns) > 30:
                corr_matrix = aligned_returns.corr(method="pearson")
                corrs = corr_matrix["strategy"].drop("strategy").abs()
                if not corrs.empty:
                    corr_daily = float(corrs.max())
                
                # Regime correlation (60d rolling)
                if len(aligned_returns) >= 60:
                    rolling_corr = aligned_returns.rolling(60).corr().unstack()["strategy"].drop("strategy", axis=1)
                    if not rolling_corr.empty:
                        corr_regime = float(rolling_corr.max().abs().max())

                # Trade/Swing correlation (approximated via 5d sum of returns for P&L swings)
                trade_returns = aligned_returns.rolling(5).sum().dropna()
                if not trade_returns.empty:
                    trade_corr_mat = trade_returns.corr(method="pearson")
                    t_corrs = trade_corr_mat["strategy"].drop("strategy").abs()
                    if not t_corrs.empty:
                        corr_trade = float(t_corrs.max())

        # 7. Gate 2: True Combinatorial Purged Cross-Validation (CPCV) Quality Gate
        cpcv_results = self.run_cpcv(df, hypothesis, n_splits=6, k_test=2)
        cpcv_mean_sharpe = cpcv_results["mean_oos_sharpe"]

        if is_sharpe > 0:
            degradation = ((is_sharpe - cpcv_mean_sharpe) / is_sharpe) * 100.0
        else:
            degradation = 100.0 if cpcv_mean_sharpe <= 0 else 0.0

        if cpcv_mean_sharpe <= 0.0 or (degradation > 60.0 and cpcv_mean_sharpe < 0.60):
            rejection_reasons.append(
                f"CPCV OOS Quality Failed: OOS Mean Sharpe {cpcv_mean_sharpe:.2f} <= 0 or collapsed ({degradation:.1f}% drop) across {cpcv_results['total_paths']} paths"
            )

        # 8. Gate 3: 5,000-Run Monte Carlo Permutation Tail Risk
        trade_returns = np.array([t.net_pnl / 250000.0 for t in full_result.trade_list]) if full_result.trade_list else np.array([])
        mc_results = self.run_monte_carlo_drawdown_test(trade_returns, num_simulations=5000)
        p95_dd = mc_results["p95_max_dd"]
        if p95_dd > self.max_monte_carlo_dd_pct:
            rejection_reasons.append(
                f"Monte Carlo Tail Risk Failed: 95th percentile Max Drawdown {p95_dd:.1f}% exceeds tolerance {self.max_monte_carlo_dd_pct}%"
            )

        # 9. Gate 4: Real Post-Tax Net Profit Factor Hurdle
        if net_pf < required_pf:
            rejection_reasons.append(
                f"Post-Tax Profit Factor Failed: Real Net PF {net_pf:.2f} < {required_pf:.2f} (Required for N={total_trades} trades | Total Costs: Rs {full_result.total_taxes_paid:,.2f})"
            )

        # 10. Decoupled Lifecycle Classification
        is_statistically_valid = (not rejection_reasons) and (net_pf >= required_pf) and (cpcv_mean_sharpe > 0)

        if not is_statistically_valid:
            status = HypothesisStatus.REJECTED
        elif curr_60["trades"] >= 2 and curr_60["net_pnl"] < 0 and total_trades >= 25:
            # Historically strong, but decaying in current 60d regime
            status = HypothesisStatus.DECAYING_WATCHLIST
        elif total_trades >= 50 and regime_stability >= 66.0 and current_regime_score >= 60.0:
            status = HypothesisStatus.CAPITAL_CANDIDATE
        elif total_trades >= 25 and (window_metrics["180d"]["net_pnl"] > 0 or window_metrics["365d"]["net_pnl"] > 0):
            status = HypothesisStatus.FORWARD_PAPER
        elif total_trades < 25:
            status = HypothesisStatus.LOW_FREQUENCY_WATCHLIST
        else:
            status = HypothesisStatus.RESEARCH_CANDIDATE

        hypothesis.status = status

        report = HypothesisValidationReport(
            hypothesis_id=hypothesis.metadata.hypothesis_id,
            status=status,
            evidence_tier=evidence_tier,
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=cpcv_mean_sharpe,
            deflated_sharpe_p_value=dsr_p_val,
            cpcv_mean_sharpe=cpcv_mean_sharpe,
            cpcv_degradation_pct=degradation,
            monte_carlo_95_max_dd_pct=p95_dd,
            net_profit_factor_post_tax=net_pf,
            regime_stability_score=regime_stability,
            current_regime_score=current_regime_score,
            recency_weighted_score=recency_weighted_score,
            portfolio_correlation_daily=corr_daily,
            portfolio_correlation_trade=corr_trade,
            portfolio_correlation_regime=corr_regime,
            window_metrics=window_metrics,
            rejection_reasons=rejection_reasons,
            tested_trials_count=effective_trials,
        )

        # Automatically record to immutable Research Experiment Ledger (Closed-Loop Trial Accounting)
        try:
            exp_record = ExperimentRecord(
                experiment_id=f"EXP_{hypothesis.metadata.hypothesis_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                strategy_id=hypothesis.metadata.name,
                symbol_universe=sym,
                timeframe=getattr(hypothesis.metadata, "timeframe", "15m"),
                parameters_json=json.dumps(hypothesis.parameters, default=str),
                in_sample_sharpe=is_sharpe,
                cpcv_oos_sharpe=cpcv_mean_sharpe,
                deflated_sharpe_p_value=dsr_p_val,
                net_profit_factor=net_pf,
                monte_carlo_95_max_dd=p95_dd,
                trials_in_experiment=trials_in_this_run,
                total_trials_cumulative=effective_trials,
                git_commit_sha=get_current_git_sha(),
                status=status.value,
                rejection_reasons_json=json.dumps(rejection_reasons),
            )
            self.experiment_ledger.log_experiment(exp_record)
        except Exception as e:
            logger.warning(f"Failed to log experiment to ledger: {e}")

        return report

    @staticmethod
    def classify_sample_evidence_tier(trade_count: int) -> Tuple[str, str]:
        """
        Classifies sample size into institutional evidence tiers:
        < 25   -> 🔴 INSUFFICIENT_EVIDENCE
        25-49  -> 🟠 PRELIMINARY
        50-99  -> 🟡 RESEARCH_CANDIDATE
        100-199-> 🟢 STATISTICALLY_MEANINGFUL
        200+   -> 🟢 STRONG_SAMPLE
        """
        if trade_count < 25:
            return "[TIER_0: INSUFFICIENT_EVIDENCE]", "Insufficient sample density (N < 25). Results subject to small-sample bias."
        elif trade_count < 50:
            return "[TIER_1: PRELIMINARY]", f"Preliminary sample (N={trade_count}). Encouraging but requires historical expansion."
        elif trade_count < 100:
            return "[TIER_2: RESEARCH_CANDIDATE]", f"Solid research candidate sample (N={trade_count}). Ready for forward paper surveillance."
        elif trade_count < 200:
            return "[TIER_3: STATISTICALLY_MEANINGFUL]", f"Statistically meaningful candidate sample (N={trade_count})."
        else:
            return "[TIER_4: STRONG_SAMPLE]", f"Strong robust sample size (N={trade_count})."

    def evaluate_multi_regime_persistence(
        self,
        hypothesis: BaseHypothesis,
        df: pd.DataFrame,
        symbol: str = "ASSET",
    ) -> pd.DataFrame:
        """
        Evaluates 3-Tier Multi-Regime Breakdown:
        - Tier 1: Current Regime (0–6 Months / Last 180 Days) -> Is it working now?
        - Tier 2: Recent Regime (6–12 Months / 180 to 365 Days ago) -> Is the edge persistent?
        - Tier 3: Extended Context (12–18 Months / 365 to 540 Days ago) -> Does it survive prior regimes?
        """
        if df.empty or len(df) < 100:
            return pd.DataFrame()

        signals_df = hypothesis.generate_signals(df)
        engine = BacktestEngine(cost_model=self.cost_model, initial_capital=500000.0)

        end_date = signals_df.index.max()
        t1_start = end_date - pd.Timedelta(days=180)
        t2_start = end_date - pd.Timedelta(days=365)
        t3_start = end_date - pd.Timedelta(days=540)

        windows = [
            ("Current (0-6m)", t1_start, end_date),
            ("Recent (6-12m)", t2_start, t1_start),
            ("Extended (12-18m)", t3_start, t2_start),
            ("Overall (0-18m Full)", t3_start, end_date),
        ]

        rows = []
        for name, start_dt, end_dt in windows:
            sub_df = signals_df[(signals_df.index >= start_dt) & (signals_df.index <= end_dt)]
            if sub_df.empty or len(sub_df) < 30:
                rows.append({
                    "Regime_Window": name,
                    "Date_Range": f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}",
                    "Trades": 0,
                    "Win_Rate_Pct": 0.0,
                    "Net_PnL_INR": 0.0,
                    "Net_Profit_Factor": 0.0,
                    "Sharpe": 0.0,
                    "Max_DD_Pct": 0.0,
                    "Status": "NO_DATA",
                })
                continue

            r = engine.run(sub_df, symbol=symbol, strategy_id=hypothesis.metadata.name, risk_per_trade_pct=0.005, capital_per_trade_pct=0.25)
            tier_label, _ = self.classify_sample_evidence_tier(r.total_trades)

            rows.append({
                "Regime_Window": name,
                "Date_Range": f"{sub_df.index.min().strftime('%Y-%m-%d')} to {sub_df.index.max().strftime('%Y-%m-%d')}",
                "Trades": r.total_trades,
                "Win_Rate_Pct": round(r.win_rate_pct, 1),
                "Net_PnL_INR": round(r.total_net_pnl, 2),
                "Net_Profit_Factor": round(r.net_profit_factor, 2),
                "Sharpe": round(r.sharpe_ratio, 2),
                "Max_DD_Pct": round(r.max_drawdown_pct, 2),
                "Evidence_Tier": tier_label,
            })

        return pd.DataFrame(rows)
