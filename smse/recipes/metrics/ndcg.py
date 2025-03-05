from typing import Any

from torch import Tensor

from smse.benchmarks.metric import Metric


class NDCG(Metric):
    """Normalized Discounted Cumulative Gain metric implementation"""

    def __init__(self, k: int = 3):
        """
        Initialize NDCG Metric.

        Args:
            k: Top ranks to consider
        """
        super().__init__(k, f"NDCG@{k}")

    def compute(
        self, predictions: Tensor, ground_truth: Tensor, indexes: Tensor, **kwargs: Any
    ) -> Tensor:
        """
        Calculate NDCG@k.

        Args:
            predictions: predict scores or rankings
            ground_truth: relevance scores / reference scores
            indexes: indices of matrix
            **kwargs: additional parameters
        Returns:
            Tensor
        """
        try:
            from torchmetrics.retrieval import RetrievalNormalizedDCG
        except ImportError:
            raise ImportError(
                "torchmetrics is not installed. Please install it using 'pip install torchmetrics'."
            )

        metric = RetrievalNormalizedDCG(top_k=self.k)
        score: Tensor = metric(predictions, ground_truth, indexes)
        return score
