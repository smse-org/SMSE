from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, k: int, name: str):
        """
        Initialize metric.

        Args:
            k: Top ranks to consider.
            name: Name of the metric.
        """
        self.k = k
        self.name = name

    @abstractmethod
    def compute(
        self, predictions: Tensor, ground_truth: Tensor, indexes: Tensor, **kwargs: Any
    ) -> Tensor:
        """
        Calculate the metric value based on targets and predictions

        Args:
            predictions: predict scores or rankings
            ground_truth: relevance scores / reference scores
            indexes: indexes of the samples
            **kwargs: additional parameters
        Returns:
            Dict containing metric name and value
        """
        pass
