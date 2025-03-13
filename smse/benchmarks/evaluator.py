from typing import Any, Dict, List
from PIL import Image

import numpy as np
import torch

from smse.benchmarks.metric import Metric
from smse.types import Modality


class Evaluator:
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

    def detect_modality(self, sample: Any) -> Modality:
        """
        Detect the modality of a given sample (TEXT, IMAGE, or AUDIO).
        
        Args:
            sample (dict, str, bytes, Image.Image, torch.Tensor, np.ndarray): 
                A dataset entry that may contain an image, audio, or text.
        
        Returns:
            Modality: The detected modality (TEXT, IMAGE, AUDIO) or None if unknown.
        """
        if isinstance(sample, dict):
            if "image" in sample:
                return Modality.IMAGE
            elif "audio" in sample:
                return Modality.AUDIO
            elif isinstance(sample.get("text", None), str):
                return Modality.TEXT
        elif isinstance(sample, str) and len(sample.split()) > 1:  
            return Modality.TEXT
        elif isinstance(sample, bytes):  
            return Modality.AUDIO
        return None


    def run(
        self, dataset: List[tuple[Any, Any]], **kwargs: Any
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

        query_modality = self.detect_modality(queries[0])
        answer_modality = self.detect_modality(relevant_answers[0])

        if query_modality is None or answer_modality is None:
            raise ValueError("Failed to infer modality from dataset samples.")

        required_modalities = {query_modality, answer_modality}
        if not required_modalities.issubset(self.model._supported_modalities):
            raise ValueError(
                f"Model does not support required modalities: {required_modalities}. "
                f"Supported modalities: {self.model._supported_modalities}"
            )

        queries_embedded = self.model.encode({query_modality: queries})[query_modality]
        relevant_answers_embedded = self.model.encode({answer_modality: relevant_answers})[answer_modality]

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