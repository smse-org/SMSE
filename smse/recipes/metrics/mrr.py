from typing import Any

from torch import Tensor

from smse.benchmarks.metric import Metric


class MRR(Metric):
    """Mean Reciprocal Rank Metric Implementation"""

    def __init__(self, k: int = 1):
        """
        Initialize MRR Metric.

        Args:
            k: Top ranks to consider.
        """
        super().__init__(k, f"MRR@{k}")

    def compute(
        self, predictions: Tensor, ground_truth: Tensor, indexes: Tensor, **kwargs: Any
    ) -> Tensor:
        """
        Calculate Mean Reciprocal Rank

        Args:
            predictions: predict scores or rankings
            ground_truth: relevance scores / reference scores
            indexes: indices of matrix
            **kwargs: additional parameters
        Returns:
            torch.Tensor: tensor with score of efficiency
        """
        try:
            from torchmetrics.retrieval import RetrievalMRR
        except ImportError:
            raise ImportError(
                "torchmetrics is not installed. Please install it using 'pip install torchmetrics'."
            )

        metric = RetrievalMRR(top_k=self.k)
        score: Tensor = metric(predictions, ground_truth, indexes)
        return score
