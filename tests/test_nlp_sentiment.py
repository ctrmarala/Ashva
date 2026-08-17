"""
Unit Tests for Financial NLP & Sentiment Engine
"""

import pytest
from src.features.nlp_sentiment import FinancialNLPEngine


def test_entity_recognition():
    engine = FinancialNLPEngine()
    
    text1 = "Reliance Industries reports 15% increase in quarterly net profit."
    entities1 = engine.extract_entities(text1)
    assert "RELIANCE" in entities1

    text2 = "TCS bags mega multi-million dollar cloud transformation deal from UK client."
    entities2 = engine.extract_entities(text2)
    assert "TCS" in entities2

    text3 = "HDFC Bank and Infosys lead benchmark Nifty 50 rally."
    entities3 = engine.extract_entities(text3)
    assert "HDFCBANK" in entities3
    assert "INFY" in entities3


def test_sentiment_scoring():
    engine = FinancialNLPEngine()

    bullish_text = "Tata Consultancy Services beats estimates with record profit surge."
    item_bullish = engine.process_article(headline=bullish_text)
    assert item_bullish.sentiment_score > 0.3
    assert item_bullish.urgency_score >= 0.2

    bearish_text = "SEBI notice issued over fraud investigation; stock downgraded with sell rating."
    item_bearish = engine.process_article(headline=bearish_text)
    assert item_bearish.sentiment_score < -0.3
    assert item_bearish.urgency_score >= 0.4
