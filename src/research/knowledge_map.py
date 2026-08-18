"""
Ashva Quantitative Research Knowledge Map & Mechanism Taxonomy
Maintains a structured, empirical registry of all explored alpha hypotheses,
tracks success/failure patterns, prevents duplicate generation, and guides orthogonal discovery.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Set
import json
from pathlib import Path


class MechanismStatus(str, Enum):
    PROVEN = "PROVEN"                   # Positive OOS PnL and multi-window stability
    EXPLORED_FAILED = "EXPLORED_FAILED" # Repeated empirical failure under statutory friction
    EXPLORED_UNCERTAIN = "EXPLORED_UNCERTAIN" # Small sample size or mixed asset results
    UNEXPLORED = "UNEXPLORED"           # High-plausibility theoretical mechanism not yet tested


class AlphaCategory(str, Enum):
    OPENING_AUCTION = "OPENING_AUCTION"
    GAP_MOMENTUM = "GAP_MOMENTUM"
    RELATIVE_STRENGTH = "RELATIVE_STRENGTH"
    STATISTICAL_REVERSION = "STATISTICAL_REVERSION"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    SECTOR_MOMENTUM = "SECTOR_MOMENTUM"
    SWING_MOMENTUM = "SWING_MOMENTUM"
    MICROSTRUCTURE_FADE = "MICROSTRUCTURE_FADE"
    ORDER_FLOW_IMBALANCE = "ORDER_FLOW_IMBALANCE"
    VOLATILITY_SQUEEZE = "VOLATILITY_SQUEEZE"
    TREND_EXHAUSTION = "TREND_EXHAUSTION"


@dataclass
class AlphaResearchRecord:
    alpha_id: str
    name: str
    category: AlphaCategory
    mechanism_description: str
    timeframe: str
    entry_window: str
    holding_concept: str
    status: MechanismStatus
    pnl_540d_inr: float
    sharpe_540d: float
    oos_trades: int
    oos_pnl_inr: float
    positive_assets: List[str] = field(default_factory=list)
    failure_lessons: str = ""
    known_limitations: str = ""


class AlphaKnowledgeMap:
    """
    Central research registry tracking explored mechanisms, empirical lessons, and unexplored frontiers.
    """

    def __init__(self):
        self.registry: Dict[str, AlphaResearchRecord] = {}
        self._load_baseline_alphas()

    def _load_baseline_alphas(self):
        """Populates the initial knowledge base from Alphas 01-31."""
        records = [
            AlphaResearchRecord(
                alpha_id="alpha_01",
                name="ALPHA_01_TRENDSURFER",
                category=AlphaCategory.SWING_MOMENTUM,
                mechanism_description="Multi-timeframe EMA trend continuation",
                timeframe="15m",
                entry_window="09:30-14:00",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=-168000,
                sharpe_540d=-3.2,
                oos_trades=180,
                oos_pnl_inr=-45000,
                failure_lessons="Generic EMA trend-following whipsaws heavily in range-bound Indian megacaps.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_02",
                name="ALPHA_02_AUCTION_ORB",
                category=AlphaCategory.OPENING_AUCTION,
                mechanism_description="Opening Range Breakout (15m ORB) with VWAP confirmation",
                timeframe="15m",
                entry_window="09:30-10:00",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=-142000,
                sharpe_540d=-2.8,
                oos_trades=210,
                oos_pnl_inr=-38000,
                failure_lessons="Standard 15m ORB has low win rate (<35%) and high false breakout rate without gap filter.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_03",
                name="ALPHA_03_VWAP_REVERSION",
                category=AlphaCategory.STATISTICAL_REVERSION,
                mechanism_description="Intraday VWAP Standard Deviation band mean reversion",
                timeframe="15m",
                entry_window="10:00-14:00",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_UNCERTAIN,
                pnl_540d_inr=-34120,
                sharpe_540d=-2.29,
                oos_trades=45,
                oos_pnl_inr=-5803,
                positive_assets=["TCS", "ICICIBANK", "AXISBANK"],
                failure_lessons="Midday VWAP reversion works modestly on low-beta banks/IT, but gets run over during trend days.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_04",
                name="ALPHA_04_GAP_AND_GO",
                category=AlphaCategory.GAP_MOMENTUM,
                mechanism_description="Large opening gap continuation with volume surge",
                timeframe="15m",
                entry_window="09:15-09:30",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_UNCERTAIN,
                pnl_540d_inr=5261,
                sharpe_540d=0.70,
                oos_trades=4,
                oos_pnl_inr=1081,
                positive_assets=["LT", "INFY", "TCS", "ICICIBANK"],
                failure_lessons="High payoff ratio (+₹5.2k, PF 99.0), but extreme rarity (N=7 total) limits statistical power.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_09",
                name="ALPHA_09_OPENING_RELATIVE_STRENGTH",
                category=AlphaCategory.RELATIVE_STRENGTH,
                mechanism_description="Opening relative strength vs cross-sectional median",
                timeframe="15m",
                entry_window="09:15-09:30",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.PROVEN,
                pnl_540d_inr=-11402,
                sharpe_540d=-0.23,
                oos_trades=111,
                oos_pnl_inr=-2262,
                positive_assets=["INFY", "TCS", "ICICIBANK", "AXISBANK", "SBIN", "LT", "RELIANCE", "BAJFINANCE"],
                failure_lessons="Broadest positive asset cluster (8/14 assets). Outstanding edge in IT (INFY +₹14.4k, TCS +₹9.6k).",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_10",
                name="ALPHA_10_STATISTICAL_RANGE_REVERSION",
                category=AlphaCategory.STATISTICAL_REVERSION,
                mechanism_description="Multi-day swing statistical range reversion holding 2-5 days",
                timeframe="15m",
                entry_window="10:00-14:30",
                holding_concept="Multi-Day Swing Delivery",
                status=MechanismStatus.PROVEN,
                pnl_540d_inr=-51181,
                sharpe_540d=-0.78,
                oos_trades=51,
                oos_pnl_inr=2581,
                positive_assets=["MARUTI", "LT", "SUNPHARMA"],
                failure_lessons="Multi-day holding eliminates forced 15:15 EOD square-off penalty, generating +₹22.6k on MARUTI.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_14",
                name="ALPHA_14_GAP_MOMENTUM_DRIFT",
                category=AlphaCategory.GAP_MOMENTUM,
                mechanism_description="Moderate opening gap (0.4-1.8 ATR) drift aligned with volume",
                timeframe="15m",
                entry_window="09:15-09:30",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.PROVEN,
                pnl_540d_inr=7743,
                sharpe_540d=0.21,
                oos_trades=131,
                oos_pnl_inr=2603,
                positive_assets=["RELIANCE", "INFY", "HDFCBANK", "LT", "TCS", "BHARTIARTL"],
                failure_lessons="Strongest overall intraday alpha. Both Long and Short profitable across 4/5 chronological windows.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_24",
                name="ALPHA_24_VOLATILITY_VACUUM_RELEASE",
                category=AlphaCategory.VOLATILITY_EXPANSION,
                mechanism_description="Midday compression followed by range/volume expansion",
                timeframe="15m",
                entry_window="11:15-13:30",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=-154659,
                sharpe_540d=-2.48,
                oos_trades=320,
                oos_pnl_inr=-42000,
                positive_assets=["TATASTEEL", "MARUTI", "HDFCBANK"],
                failure_lessons="Midday breakouts suffer from low volume and high false breakout chop in Indian megacaps.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_25",
                name="ALPHA_25_CROSS_SECTIONAL_RESIDUAL_REVERSION",
                category=AlphaCategory.STATISTICAL_REVERSION,
                mechanism_description="Fading idiosyncratic moves when market index is calm",
                timeframe="15m",
                entry_window="10:00-14:00",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=-133950,
                sharpe_540d=-3.73,
                oos_trades=160,
                oos_pnl_inr=-39000,
                failure_lessons="Idiosyncratic divergence is flow MOMENTUM, not noise. Fading divergence yields 28% win rate.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_26",
                name="ALPHA_26_RELATIVE_STRENGTH_PERSISTENCE",
                category=AlphaCategory.RELATIVE_STRENGTH,
                mechanism_description="Multi-bar relative strength persistence outside opening",
                timeframe="15m",
                entry_window="10:00-13:30",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=0,
                sharpe_540d=0,
                oos_trades=0,
                oos_pnl_inr=0,
                failure_lessons="Step 0 diagnostic proved midday RS gross edge is <1.5 bps, failing the 7.0 bps statutory tax hurdle.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_27",
                name="ALPHA_27_SECTOR_MOMENTUM_DRIFT",
                category=AlphaCategory.SECTOR_MOMENTUM,
                mechanism_description="Synchronous sector momentum breakout",
                timeframe="15m",
                entry_window="09:45-13:30",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=-168170,
                sharpe_540d=-6.74,
                oos_trades=141,
                oos_pnl_inr=-18648,
                positive_assets=["INFY", "TCS"],
                failure_lessons="IT sector co-moves cleanly (+₹9.8k INFY), but domestic banks suffer severe pair-rotation chop.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_28",
                name="ALPHA_28_VALUE_AREA_EXPANSION",
                category=AlphaCategory.OPENING_AUCTION,
                mechanism_description="Breakout outside the morning 70% Volume Profile Value Area",
                timeframe="15m",
                entry_window="11:15-13:30",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=-103159,
                sharpe_540d=-5.07,
                oos_trades=167,
                oos_pnl_inr=-5717,
                positive_assets=["INFY", "HDFCBANK"],
                failure_lessons="Midday Value Area boundaries in large caps act as mean-reverting liquidity magnets back to POC.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_29",
                name="ALPHA_29_TREND_EXHAUSTION_CLIMAX",
                category=AlphaCategory.TREND_EXHAUSTION,
                mechanism_description="Fading vertical price/volume spikes over-extended from VWAP",
                timeframe="15m",
                entry_window="09:30-14:00",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=-24912,
                sharpe_540d=-3.73,
                oos_trades=12,
                oos_pnl_inr=-7612,
                positive_assets=["SUNPHARMA", "BHARTIARTL"],
                failure_lessons="Climax spikes without multi-day regime context frequently overshoot stops before reverting to VWAP.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_30",
                name="ALPHA_30_MIDDAY_SQUEEZE_TREND",
                category=AlphaCategory.VOLATILITY_SQUEEZE,
                mechanism_description="30m Bollinger Band compression inside Keltner Channels",
                timeframe="30m",
                entry_window="09:45-14:00",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=-226191,
                sharpe_540d=-5.88,
                oos_trades=341,
                oos_pnl_inr=-67915,
                failure_lessons="Midday squeeze breakouts in Indian large-caps fire dozens of false expansions in sideways consolidation.",
            ),
            AlphaResearchRecord(
                alpha_id="alpha_31",
                name="ALPHA_31_FAILED_OPENING_DRIVE_FADE",
                category=AlphaCategory.MICROSTRUCTURE_FADE,
                mechanism_description="Fading failed opening drives with long rejection wicks",
                timeframe="15m",
                entry_window="09:30",
                holding_concept="Intraday 15:15 Square-off",
                status=MechanismStatus.EXPLORED_FAILED,
                pnl_540d_inr=-11770,
                sharpe_540d=-2.22,
                oos_trades=19,
                oos_pnl_inr=-5124,
                positive_assets=["RELIANCE", "BAJFINANCE", "MARUTI"],
                failure_lessons="Fading rejection wicks works on true traps, but suffers catastrophic losses when opening drives have real trend momentum.",
            ),
        ]
        for r in records:
            self.registry[r.alpha_id] = r

    def is_novel_hypothesis(self, category: AlphaCategory, mechanism_desc: str, timeframe: str, entry_window: str) -> bool:
        """
        Determines if a candidate hypothesis is structurally novel or merely a duplicate.
        """
        for r in self.registry.values():
            if (r.category == category and r.timeframe == timeframe and r.entry_window == entry_window):
                return False
        return True

    def get_explored_categories(self) -> Dict[AlphaCategory, int]:
        counts: Dict[AlphaCategory, int] = {}
        for r in self.registry.values():
            counts[r.category] = counts.get(r.category, 0) + 1
        return counts

    def get_unexplored_mechanisms(self) -> List[Dict[str, Any]]:
        """
        Identifies high-plausibility research territory that has not yet been saturated or failed.
        """
        candidate_territory = [
            {
                "proposed_id": "alpha_32",
                "category": AlphaCategory.ORDER_FLOW_IMBALANCE,
                "name": "ALPHA_32_VOLUME_WEIGHTED_CLOSING_IMBALANCE",
                "mechanism_description": "Institutional MOC (Market-On-Close) & Power Hour Volume Drift (14:15 to 15:15 IST)",
                "timeframe": "15m",
                "entry_window": "14:15-14:30",
                "holding_concept": "Power Hour Drift into 15:15 close",
                "economic_rationale": (
                    "Passive index funds, institutional rebalancing flows, and structured product hedging execute "
                    "in the final trading hour (14:15-15:15 IST). When an asset exhibits a sharp volume spike "
                    "at 14:15 breaking the 14:00 high/low, institutional flow tends to persist into the official close."
                ),
            },
            {
                "proposed_id": "alpha_33",
                "category": AlphaCategory.SWING_MOMENTUM,
                "name": "ALPHA_33_MULTI_DAY_CONSOLIDATION_BREAKOUT",
                "mechanism_description": "Multi-day range contraction with multi-session holding horizon (2 to 5 days)",
                "timeframe": "60m",
                "entry_window": "14:00-15:00",
                "holding_concept": "Multi-Day Overnight Swing",
                "economic_rationale": (
                    "Following 3+ days of contracting daily ranges (NR3/NR4), a breakout on 60m bars that closes outside "
                    "the 3-day high/low initiates a multi-day trend. Holding across 2-5 days amortizes round-trip Indian taxes."
                ),
            },
        ]
        return candidate_territory

    def register_experiment_result(self, record: AlphaResearchRecord):
        """Adds a newly tested alpha to the knowledge base."""
        self.registry[record.alpha_id] = record
