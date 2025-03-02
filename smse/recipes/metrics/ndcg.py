from smse.benchmarks.metric import Metric
from sklearn.metrics import ndcg_score
from typing import Dict

import numpy as np

class NDCG(Metric):
    """Normalized Discounted Cumulative Gain metric implementation"""

    def __init__(self, k: int = 10, name: str = None):
        name = f"NDCG@{k}" if name is None else name
        super().__init__(name)
        self.k = k

    def compute(self, y_target: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Calculate NDCG@k.

        Args:
            y_target: relevance scores for each items
            y_pred: predicted scores

        Returns:
            Dict with NDCG scores
        """
        score = ndcg_score(y_target, y_pred, k=self.k)
        return {self.name: score}