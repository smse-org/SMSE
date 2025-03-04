from smse.benchmarks.metric import Metric
from scipy.stats import spearmanr
from typing import Dict

import numpy as np

class SpearmanCorrelation(Metric):
    """Spearman's rank correlation coefficient metric."""

    def __init__(self, name: str = "Spearman"):
        super().__init__(name)

    def compute(self, y_target: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Calculate Spearman's rank correlation using scipy

        Args:
            y_target: ground truth values or ranks
            y_pred: predict values or# Get top n predictions for each quer}

        Returns:
            Dict with Spearman correlation coefficient
        """
        if np.std(y_target.cpu().numpy()) == 0 or np.std(y_pred.cpu().numpy()) == 0:
            return {self.name: 0.0}

        corr, _ = spearmanr(y_target, y_pred)
        return {self.name: corr if not np.isnan(corr).any() else 0.0}