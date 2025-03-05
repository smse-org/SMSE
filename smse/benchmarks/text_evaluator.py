from typing import Any, Dict, List

import torch

from smse.benchmarks.metric import Metric
from smse.types import Modality


class TextEvaluator:
    """Evaluation framework for text models."""

    def __init__(self, model: Any, metrics: List[Metric]):
        """
        Initialize evaluator

        Args:
            model: Model object that makes predictions
            metrics: List of Metric objects to evaluate
        """
        self.model = model
        self.metrics = metrics

    def cosine_to_rankings(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Converts a cosine similarity matrix into ranking-based relevance scores
        while keeping the original positions.

        For each query (row), the most similar document gets rank 1, the second most similar gets rank 2, and so on.

        Args:
            similarity_matrix (torch.Tensor): The cosine similarity matrix (queries x documents).
        Returns:
            torch.Tensor: A tensor with the same shape as similarity_matrix, where each entry represents the rank
                        of the corresponding document for that query.
        """
        rankings = similarity_matrix.argsort(
            dim=1, descending=True
        )  # Get sorted indices
        ranked_matrix = torch.zeros_like(
            rankings, dtype=torch.float
        )  # Initialize result tensor

        # Assign ranks while keeping original document positions
        for i in range(rankings.shape[0]):
            ranked_matrix[i, rankings[i]] = torch.arange(
                1, rankings.shape[1] + 1, dtype=torch.float
            )

        return ranked_matrix

    def run(
        self, dataset: List[tuple[str, str]], **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Run evaluation using specificed metrics

        Args:
            dataset: the dataset holding for each query its corresponding answer.
            **kwargs: Additional parameters for metrics
        Returns:
            results
        """
        ground_truth = torch.eye(len(dataset))

        queries_raw, relevant_answers_raw = zip(*dataset)
        queries = list(queries_raw)
        relevant_answers = list(relevant_answers_raw)

        queries_embedded = self.model.encode({Modality.TEXT: queries})[Modality.TEXT]
        relevant_answers_embedded = self.model.encode(
            {Modality.TEXT: relevant_answers}
        )[Modality.TEXT]

        similarity_matrix = self.model.similarity(
            queries_embedded, relevant_answers_embedded
        )
        num_queries, num_docs = similarity_matrix.shape
        indexes = torch.arange(num_docs).repeat(
            num_queries, 1
        )  # Ensure each row contains proper document indices

        results = {}

        for metric in self.metrics:
            try:
                metric_res = metric.compute(
                    similarity_matrix, ground_truth, indexes, **kwargs
                )
            except Exception as e:
                raise RuntimeError(f"Error comupting metric {metric.name}: {e}") from e
            results[metric.name] = metric_res

        return results
