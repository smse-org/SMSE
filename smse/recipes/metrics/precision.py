from typing import Any

from torch import Tensor

from smse.benchmarks.metric import Metric


class Precision(Metric):
    """Precision Metric Implementation"""

    def __init__(self, k: int = 1):
        """
        Initialize Precision Metric.

        Args:
            k: Top ranks to consider.
        """
        super().__init__(k, f"Precision@{k}")

    def compute(
        self, predictions: Tensor, ground_truth: Tensor, indexes: Tensor, **kwargs: Any
    ) -> Tensor:
        """
        Calculate Precision Metric

        Args:
            predictions: predict scores or rankings
            ground_truth: relevance scores / reference scores
            indexes: indices of matrix
            **kwargs: additional parameters
        Returns:
            torch.Tensor: tensor with score of efficiency
        """
        try:
            from torchmetrics.retrieval import RetrievalPrecision
        except ImportError:
            raise ImportError(
                "torchmetrics is not installed. Please install it using 'pip install torchmetrics'."
            )

        metric = RetrievalPrecision(top_k=self.k)
        score: Tensor = metric(predictions, ground_truth, indexes)
        return score
