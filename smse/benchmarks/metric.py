from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, y_target: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Calculate the metric value based on targets and predictions

        Args:
            y_target: relevance scores / reference scores
            y_pred: predict scores or rankings

        Returns:
            Dict containing metric name and value
        """
        pass