"""
Ashva Financial NLP & News Sentiment Engine
Parses corporate announcements, RSS financial feeds, and headlines to generate entity-tagged sentiment alpha scores.
"""

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Dict, List, Optional, Tuple, Any


@dataclass(frozen=True)
class NewsItem:
    headline: str
    body: str
    source: str
    timestamp: datetime
    matched_symbols: List[str]
    sentiment_score: float  # -1.0 (Extreme Bearish) to +1.0 (Extreme Bullish)
    urgency_score: float    # 0.0 to 1.0


class FinancialNLPEngine:
    """
    NLP sentiment scoring and entity recognition tailored for Indian financial markets (NSE/BSE).
    """

    # Common Indian ticker mapping dictionaries
    TICKER_ALIASES = {
        "RELIANCE": ["RELIANCE", "RELIANCE INDUSTRIES", "RIL", "JIO", "RELIANCE RETAIL"],
        "TCS": ["TCS", "TATA CONSULTANCY", "TATA CONSULTANCY SERVICES"],
        "HDFCBANK": ["HDFC BANK", "HDFC", "HOUSING DEVELOPMENT FINANCE"],
        "INFY": ["INFOSYS", "INFY"],
        "ICICIBANK": ["ICICI", "ICICI BANK"],
        "SBIN": ["SBI", "STATE BANK OF INDIA"],
        "TATAMOTORS": ["TATA MOTORS", "JLR", "JAGUAR LAND ROVER"],
        "LT": ["L&T", "LARSEN", "LARSEN & TOUBRO"],
        "BHARTIARTL": ["AIRTEL", "BHARTI AIRTEL"],
        "ITC": ["ITC", "ITC LTD"],
    }

    # Domain financial lexicon with calibrated weights
    BULLISH_KEYWORDS = {
        "profit surges": 0.85, "record profit": 0.90, "beats estimates": 0.80,
        "revenue jumps": 0.75, "order win": 0.70, "bags contract": 0.75,
        "expansion": 0.50, "upgrade": 0.65, "dividend hike": 0.60,
        "growth": 0.40, "acquisition": 0.50, "usfda approval": 0.85,
        "positive": 0.40, "strong earnings": 0.80, "bullish": 0.60,
        "buy rating": 0.70, "guidance raised": 0.85,
    }

    BEARISH_KEYWORDS = {
        "profit drops": -0.85, "losses widen": -0.90, "misses estimates": -0.80,
        "revenue falls": -0.75, "probe": -0.70, "fraud": -0.95,
        "sebi notice": -0.85, "downgrade": -0.65, "default": -0.95,
        "debt crisis": -0.85, "resignation": -0.45, "usfda warning": -0.85,
        "negative": -0.40, "weak earnings": -0.80, "bearish": -0.60,
        "sell rating": -0.70, "guidance slashed": -0.85, "penalty": -0.65,
    }

    HIGH_URGENCY_KEYWORDS = [
        "breaking", "urgent", "usfda", "sebi", "earnings", "results", "quarterly", "contract", "fraud", "merger"
    ]

    def extract_entities(self, text: str) -> List[str]:
        """
        Extracts matching NSE stock tickers from headline or news text.
        """
        text_upper = text.upper()
        matched = []

        for symbol, aliases in self.TICKER_ALIASES.items():
            for alias in aliases:
                # Word boundary match
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, text_upper):
                    matched.append(symbol)
                    break
        return list(set(matched))

    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Calculates sentiment polarity [-1.0, +1.0] and urgency score [0.0, 1.0].
        """
        text_lower = text.lower()
        score = 0.0
        hits = 0

        for kw, weight in self.BULLISH_KEYWORDS.items():
            if kw in text_lower:
                score += weight
                hits += 1

        for kw, weight in self.BEARISH_KEYWORDS.items():
            if kw in text_lower:
                score += weight
                hits += 1

        # Normalized sentiment score
        if hits > 0:
            final_sentiment = max(-1.0, min(1.0, score / hits))
        else:
            final_sentiment = 0.0  # Neutral

        # Urgency calculation
        urgency = 0.2  # Base urgency
        for kw in self.HIGH_URGENCY_KEYWORDS:
            if kw in text_lower:
                urgency += 0.25
        final_urgency = min(1.0, urgency)

        return round(final_sentiment, 3), round(final_urgency, 2)

    def process_article(
        self,
        headline: str,
        body: str = "",
        source: str = "RSS",
        timestamp: Optional[datetime] = None,
    ) -> NewsItem:
        """
        End-to-end parsing of a single news article into a structured NewsItem.
        """
        ts = timestamp or datetime.now()
        full_text = f"{headline} {body}"
        matched_symbols = self.extract_entities(full_text)
        sentiment, urgency = self.analyze_sentiment(full_text)

        return NewsItem(
            headline=headline,
            body=body,
            source=source,
            timestamp=ts,
            matched_symbols=matched_symbols,
            sentiment_score=sentiment,
            urgency_score=urgency,
        )
