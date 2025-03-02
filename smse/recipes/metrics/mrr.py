from smse.benchmarks.metric import Metric
from typing import Dict

import numpy as np

class MRR(Metric):
    """Mean Reciprocal Rank Metric Implementation"""

    def __init__(self, name: str = "MRR"):
        super().__init__(name)

    def compute(self, y_target: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Calculate Mean Reciprocal Rank

        Args:
            y_target: binary relevance labels (1 for relevant, 0 for irrelevant)
            y_pred: predict scores or probabilities

        Returns:
            Dict with MRR score
        """
        # Get rankings based on predicted scores
        rankings = (-y_pred).argsort()

        # Find pos of first relevant item for each query
        reciprocal_ranks = []

        for i, (true, ranks) in enumerate(zip(y_target, rankings)):
            relevant_pos = np.where(true[ranks] == 1)[0]

            if len(relevant_pos) == 0:
                reciprocal_ranks.append(0.0)
                continue

            for rank_idx, item_idx in enumerate(ranks):
                if item_idx in relevant_pos:
                    reciprocal_ranks.append(1.0 / (rank_idx + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)

        mrr = np.mean(reciprocal_ranks) if len(reciprocal_ranks) > 0 else 0.0
        return {self.name: mrr}