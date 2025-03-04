from smse.benchmarks.metric import Metric
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
            y_target: relevance scores for each item
            y_pred: predict scores or rankings

        Returns:
            Dict with Top-N accuracy score
        """
        # Get top n predictions for each query
        top_k_indicies = np.argsort(-y_pred, axis = 1)[:, :self.k]

        # Check if any relevant items are in top n
        hits = 0
        for i, (true, indices) in enumerate(zip(y_target, top_k_indicies)):
            if np.sum(true[indices]) > 0:
                hits += 1

        accuracy = hits / len(y_target) if len(y_target) > 0 else 0.0
        return {self.name: accuracy}