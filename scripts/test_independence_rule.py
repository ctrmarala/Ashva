import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.research.validator import StatisticalValidator
from src.research.hypothesis_factory import HypothesisFactory
from src.research.mechanisms.time_of_day import BaseTimeOfDayHypothesis

def main():
    print("Testing Portfolio Independence Validator & Market Mechanism Factory...")
    
    # 1. Create a dummy dataframe
    dates = pd.date_range("2025-01-01", "2025-06-30", freq="15min")
    # Filter for market hours (09:15 to 15:30)
    market_hours = (dates.hour >= 9) & (dates.hour <= 15)
    dates = dates[market_hours]
    
    # Generate random walk prices
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, len(dates))
    prices = 1000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.001,
        "low": prices * 0.999,
        "close": prices * (1 + np.random.normal(0, 0.0005, len(dates))),
        "volume": np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # 2. Create a dummy baseline portfolio (e.g., highly correlated random stream)
    # The factory takes daily returns
    daily_dates = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    
    # Let's mock a strategy return stream and make baseline identical to force correlation
    baseline_returns = pd.DataFrame({
        "Alpha01": np.random.normal(0, 0.01, len(daily_dates))
    }, index=daily_dates)
    
    # We will pass this to the validator, but we need to ensure the strategy's returns 
    # match baseline. In this script, it's a random walk, so we will just set correlation limit to 0.00
    validator = StatisticalValidator(max_portfolio_correlation=0.0001)
    factory = HypothesisFactory(validator=validator, baseline_portfolio_returns=baseline_returns)
    
    print("\nRunning HypothesisFactory on BaseTimeOfDayHypothesis...")
    reports = factory.evaluate_hypothesis_space(
        hypothesis_cls=BaseTimeOfDayHypothesis,
        metadata=None,
        df=df,
        custom_param_grid={"morning_trend_threshold_pct": [0.5]}
    )
    
    for report in reports:
        print(f"\nHypothesis: {report.hypothesis_id}")
        print(f"Status: {report.status}")
        print(f"Portfolio Correlation: {report.portfolio_correlation:.3f}")
        print("Rejection Reasons:", report.rejection_reasons)

if __name__ == "__main__":
    main()
