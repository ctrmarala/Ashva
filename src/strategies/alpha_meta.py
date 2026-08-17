"""
Alpha Strategy 3: Machine Learning Meta-Labeling Ensemble
Implements Marcos López de Prado's Meta-Labeling technique:
Uses a secondary ML model to predict primary strategy win probabilities and dynamically size / filter bets.
Enforces strict separation of training (fit_meta_model) and out-of-sample inference to prevent lookahead leakage.
"""

from typing import Dict, List, Any, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.strategies.base import BaseStrategy
from src.research.hypothesis import BaseHypothesis, HypothesisMetadata
from src.research.triple_barrier import TripleBarrierLabeler
from src.features.microstructure import MicrostructureFeatureExtractor
from src.core.events import BarEvent, SignalEvent, EventType

logger = logging.getLogger("Ashva.AlphaMeta")


class AlphaMetaLabeledStrategy(BaseStrategy, BaseHypothesis):
    """
    Hypothesis 3: ML Meta-Labeled Primary Strategy Filter.
    """

    DEFAULT_METADATA = HypothesisMetadata(
        hypothesis_id="ALPHA_03_META_LABELED_ENSEMBLE",
        name="Machine Learning Meta-Labeled Alpha Filter & Dynamic Sizer",
        category="META_LABELING",
        economic_rationale=(
            "Primary heuristic signals generate both true and false breakouts. "
            "A secondary machine learning model trained on multi-factor microstructure features predicts "
            "the probability of hitting the profit-taking barrier before the stop-loss, filtering low-conviction trades."
        ),
        target_instruments=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        timeframe="5m",
    )

    def __init__(
        self,
        primary_strategy: BaseStrategy,
        metadata: Optional[HypothesisMetadata] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        meta = metadata or self.DEFAULT_METADATA
        BaseHypothesis.__init__(self, metadata=meta, parameters=parameters)
        BaseStrategy.__init__(self, strategy_id=meta.hypothesis_id, parameters=parameters)

        self.primary_strategy = primary_strategy
        self.min_conviction_threshold = self.parameters.get("min_conviction_threshold", 0.55)
        self.n_estimators = self.parameters.get("n_estimators", 50)
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=4, random_state=42)
        self.is_fitted = False
        self._live_bars: List[BarEvent] = []

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            "min_conviction_threshold": [0.50, 0.55, 0.60],
            "n_estimators": [30, 50, 100],
        }

    def _extract_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts microstructure and trend features for meta-model training."""
        extractor = MicrostructureFeatureExtractor()
        df_feat = extractor.calculate_anchored_vwap(df)
        df_feat = extractor.calculate_volume_delta(df_feat)

        features = pd.DataFrame(index=df.index)
        features["vwap_dist"] = (df_feat["close"] - df_feat["vwap"]) / df_feat["vwap"].replace(0, np.nan)
        features["vol_surge"] = df_feat["volume_surge_ratio"]
        features["cvd_norm"] = df_feat["cvd"] / (df_feat["volume"].rolling(50).sum() + 1e-8)
        features["volatility"] = df["close"].pct_change().rolling(20).std().fillna(0.0)

        return features.fillna(0.0)

    def fit_meta_model(self, df_train: pd.DataFrame):
        """
        Explicitly trains the meta-model on primary signals and triple-barrier outcomes from training data.
        Must be called prior to out-of-sample inference.
        """
        # 1. Generate primary signals on training slice
        primary_df = self.primary_strategy.generate_signals(df_train)
        
        # 2. Compute Triple-Barrier Outcomes
        barrier_outcomes = TripleBarrierLabeler.apply_triple_barrier(
            df=primary_df,
            pt_mult=1.5,
            sl_mult=1.0,
            max_holding_bars=15,
        )

        if barrier_outcomes.empty or len(barrier_outcomes) < 5:
            logger.warning("Insufficient samples to train Meta-Labeler. Defaulting to unfitted baseline.")
            self.is_fitted = True
            return

        # 3. Align Features to Trade Entry Times
        features_df = self._extract_meta_features(primary_df)
        
        entry_times = barrier_outcomes["entry_time"].values
        X = features_df.loc[entry_times].values
        y = (barrier_outcomes["label"].values == 1).astype(int)

        if len(np.unique(y)) > 1:
            self.model.fit(X, y)
            logger.info(f"Meta-Labeler trained successfully on {len(X)} trade samples.")
        self.is_fitted = True

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters and sizes primary strategy signals based on pre-trained win probability.
        Will NOT silently train on inference data.
        """
        primary_df = self.primary_strategy.generate_signals(df)
        
        if not self.is_fitted:
            logger.warning("AlphaMetaLabeledStrategy evaluated without pre-training! Forwarding primary signals.")
            return primary_df

        features_df = self._extract_meta_features(primary_df)
        primary_signals = primary_df["signal"].values
        final_signals = np.zeros(len(df))

        for i in range(len(df)):
            p_sig = primary_signals[i]
            if p_sig == 0.0:
                continue

            if self.is_fitted and hasattr(self.model, "classes_") and len(self.model.classes_) > 1:
                feat_vector = features_df.iloc[i].values.reshape(1, -1)
                win_prob = self.model.predict_proba(feat_vector)[0][1]
            else:
                win_prob = 0.60  # Default baseline

            if win_prob >= self.min_conviction_threshold:
                sizing_factor = float(np.clip(2.0 * win_prob - 1.0, 0.2, 1.0))
                final_signals[i] = p_sig * sizing_factor
            else:
                final_signals[i] = 0.0

        primary_df["signal"] = final_signals
        return primary_df

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """Live streaming incremental bar handler."""
        self._live_bars.append(bar)
        if len(self._live_bars) > 200:
            self._live_bars.pop(0)

        # Delegate to primary strategy
        primary_signal = self.primary_strategy.on_bar(bar)
        if not primary_signal or primary_signal.direction == 0.0:
            return None

        if not self.is_fitted or not hasattr(self.model, "classes_") or len(self.model.classes_) <= 1:
            return primary_signal

        # Extract live features
        df_live = pd.DataFrame([
            {
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in self._live_bars
        ], index=[b.timestamp for b in self._live_bars])

        feat_df = self._extract_meta_features(df_live)
        feat_vector = feat_df.iloc[-1].values.reshape(1, -1)
        win_prob = float(self.model.predict_proba(feat_vector)[0][1])

        if win_prob >= self.min_conviction_threshold:
            sizing = float(np.clip(2.0 * win_prob - 1.0, 0.2, 1.0))
            return SignalEvent(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=primary_signal.direction,
                strength=sizing,
                strategy_id=self.strategy_id,
                stop_loss=primary_signal.stop_loss,
                take_profit=primary_signal.take_profit,
                metadata={
                    "ml_conviction": win_prob,
                    "primary_strategy": self.primary_strategy.strategy_id,
                },
            )
        else:
            return None
