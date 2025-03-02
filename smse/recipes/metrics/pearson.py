from smse.benchmarks.metric import Metric
from scipy.stats import pearsonr
from typing import Dict

import numpy as np

class PearsonCorrelation(Metric):
    """Pearson's correlation coefficient metric."""

    def __init__(self, name: str = "Pearson"):
        super().__init__(name)

    def compute(self, y_target: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Calculate Pearson's correlation using scipy

        Args:
            y_target: ground truth values or ranks
            y_pred: predict values or# Get top n predictions for each quer}

        Returns:
            Dict with Pearson correlation coefficient
        """
        if np.std(y_target) == 0 or np.std(y_pred) == 0:
            return {self.name: 0.0}

        corr = pearsonr(y_target, y_pred)[0]
        return {self.name: corr if not np.isnan(corr) else 0.0}