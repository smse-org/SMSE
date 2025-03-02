from smse.benchmarks.metric import Metric
from typing import Dict

import numpy as np

class Recall(Metric):
    """Recall metric implementation"""

    def __init__(self, k: int = 10, name: str = None):
        name = f"Recall@{k}" if name is None else name
        super().__init__(name)
        self.k = k

    def compute(self, y_target: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Calculate Recall@k.

        Args:
            y_target: binary relevance labels (1 for relevant, 0 for irrelevant)
            y_pred: predict scores or rankings

        Returns:
            Dict with recall score
        """
        # Get top n predictions for each query
        top_k_indicies = np.argsort(-y_pred, axis = 1)[:, :self.k]

        recall_vals = []
        for i, (true, indices) in enumerate(zip(y_target, top_k_indicies)):
            # Number of relevant items in top k
            n_relevant_in_top_k = np.sum(true[indices])
            n_relevant = np.sum(true)

            if n_relevant > 0:
                recall_vals.append(n_relevant_in_top_k / n_relevant)
            else:
                recall_vals.append(1.0) # Perfect recall if no relevant items

        avg_recall = np.mean(recall_vals) if len(recall_vals) > 0 else 0.0
        return {self.name: avg_recall}