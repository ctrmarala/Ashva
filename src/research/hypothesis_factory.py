"""
Ashva Quantitative Hypothesis Factory & Research Lab
Systematically parameterizes, backtests, and validates multi-asset hypotheses across parameter spaces.
"""

from itertools import product
from typing import List, Dict, Any, Type
import pandas as pd

from src.research.hypothesis import BaseHypothesis, HypothesisMetadata, HypothesisStatus, HypothesisValidationReport
from src.research.validator import StatisticalValidator


class HypothesisFactory:
    """
    Automated factory to instantiate, parameterize, test, and filter hypotheses.
    """

    def __init__(self, validator: StatisticalValidator = None, baseline_portfolio_returns: pd.DataFrame = None):
        self.validator = validator or StatisticalValidator()
        self.accepted_hypotheses: List[Dict[str, Any]] = []
        self.rejected_hypotheses: List[Dict[str, Any]] = []
        self.baseline_portfolio_returns = baseline_portfolio_returns if baseline_portfolio_returns is not None else pd.DataFrame()

    def generate_parameter_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Expands a dictionary of parameter lists into a Cartesian product list of parameter dictionaries.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations

    def evaluate_hypothesis_space(
        self,
        hypothesis_cls: Type[BaseHypothesis],
        metadata: HypothesisMetadata,
        df: pd.DataFrame,
        custom_param_grid: Dict[str, List[Any]] = None,
    ) -> List[HypothesisValidationReport]:
        """
        Generates and tests all parameter permutations of a given hypothesis class against input data.
        """
        # Create prototype instance to extract default param grid
        proto = hypothesis_cls(metadata=metadata)
        param_grid = custom_param_grid or proto.get_parameter_grid()
        combinations = self.generate_parameter_combinations(param_grid)
        num_trials = len(combinations)

        reports = []

        for i, params in enumerate(combinations):
            hyp_instance = hypothesis_cls(metadata=metadata, parameters=params)
            report = self.validator.validate_hypothesis(
                hypothesis=hyp_instance,
                df=df,
                num_trials=num_trials,
                baseline_portfolio_returns=self.baseline_portfolio_returns,
            )
            
            record = {
                "instance": hyp_instance,
                "parameters": params,
                "report": report,
            }

            if report.is_accepted():
                self.accepted_hypotheses.append(record)
            else:
                self.rejected_hypotheses.append(record)

            reports.append(report)

        return reports

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Returns a tabular comparison DataFrame of all tested hypotheses and their rejection/acceptance status.
        """
        all_records = self.accepted_hypotheses + self.rejected_hypotheses
        rows = []
        for item in all_records:
            rep = item["report"]
            row = rep.to_dict()
            row["parameters"] = str(item["parameters"])
            rows.append(row)
        return pd.DataFrame(rows)
