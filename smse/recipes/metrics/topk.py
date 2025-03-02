from smse.benchmarks.metric import Metric
from sklearn.metrics import top_k_accuracy_score
from typing import Dict

import numpy as np

class TopK(Metric):
    """Top-K accuracy metric implementation with zero-shot as standard"""

    def __init__(self, k: int = 1, name: str = None):
        name = f"Top-{k}" if name is None else name
        super().__init__(name)
        self.k = k

    def compute(self, y_target: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Calculate Top-K accuracy

        Args:
            y_target: binary relevance labels (1 for relevant, 0 for irrelevant)
            y_pred: predict scores or rankings

        Returns:
            Dict with Top-K accuracy score
        """
        # Get top n predictions for each query
        try:
            score = top_k_accuracy_score(y_target, y_pred, k=self.k)
            return {self.name: score}
        except Exception as e:
            top_k_indicies = np.argsort(-y_pred, axis = 1)[:, :self.k]

            # Check if any relevant items are in top n
            hits = 0
            for i, (true, indices) in enumerate(zip(y_target, top_k_indicies)):
                if np.sum(true[indices]) > 0:
                    hits += 1

            accuracy = hits / len(y_target) if len(y_target) > 0 else 0.0
            return {self.name: accuracy}