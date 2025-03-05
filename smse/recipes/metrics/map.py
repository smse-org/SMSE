from typing import Any

from torch import Tensor

from smse.benchmarks.metric import Metric


class MAP(Metric):
    """MAP accuracy metric implementation"""

    def __init__(self, k: int = 3):
        """
        Initialize MAP Metric.

        Args:
            k: Top ranks to consider.
        """
        super().__init__(k, f"MAP@{k}")

    def compute(
        self, predictions: Tensor, ground_truth: Tensor, indexes: Tensor, **kwargs: Any
    ) -> Tensor:
        """
        Calculate Mean Average Percision accuracy

        Args:
            predictions: predict scores or rankings
            ground_truth: relevance scores / reference scores
            indexes: indices of matrix
            **kwargs: additional parameters
        Returns:
            Tensor
        """
        try:
            from torchmetrics.retrieval import RetrievalMAP
        except ImportError:
            raise ImportError(
                "torchmetrics is not installed. Please install it using 'pip install torchmetrics'."
            )

        metric = RetrievalMAP(top_k=self.k)
        score: Tensor = metric(predictions, ground_truth, indexes)
        return score
