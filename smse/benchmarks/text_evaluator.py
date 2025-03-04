from smse.benchmarks.metric import Metric
from smse.types import Modality
from typing import Dict, List, Any

import numpy as np

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
        self.results = {}

    def cosine_to_relevance_quantiles(similarites, bin=4):
        """
        Converts cosine similarity matrix to relevance quantiles so it can be used
        as predictions for comparison.

        Args:
            similarities: the cosine similarity matrix to be converted.
            bin: number of bins to convert to.
        Returns:
            relevance_scores: similarity matrix converted to prediction.
        """
        flattened = similarites.flatten() # Flatten to get all values
        thresholds = np.percentile(flattened, np.linspace(0, 100, bins+1))

        relevance_scores = np.digitize(similarites, thresholds, right=True) - 1
        return relevance_scores

    def create_ground_truth(self, dataset: List[tuple[str, str]]) -> np.ndarray:
        """
        Converts dataset into a ground truth table, for each query its corresponding 
        correct answer marked as 1, otherwise 0.

        Args:
            dataset: the dataset holding for each query its corresponding answer.
        Returns:
            ground_truth: ground truth table constructed from dataset
        """
        queries = list(set(q for q, _ in dataset))
        answers = list(set(a for _, a in dataset))

        query_to_index = {q: i for i, q in enumerate(queries)}
        ans_to_index = {d: i for i, d in enumerate(answers)}

        ground_truth = np.zeros((len(queries), len(answers)))

        for query, answer in dataset:
            q_idx = query_to_index[query]
            a_idx = ans_to_index[answer]
            ground_truth[q_idx, a_idx] = 1

        return ground_truth

    def run(self, dataset: List[tuple[str, str]], **kwargs):
        """
        Run evaluation using specificed metrics

        Args:
            dataset: the dataset holding for each query its corresponding answer.
            **kwargs: Additional parameters for metrics
        """
        ground_truth = self.create_ground_truth(dataset)

        queries, relevant_answers = zip(*dataset)
        queries = list(queries)
        relevant_answers = list(relevant_answers)

        queries = self.model.encode({Modality.TEXT: queries})[Modality.TEXT]
        relevant_answers = self.model.encode({Modality.TEXT: queries})[Modality.TEXT]

        similarity_matrix = self.model.similarity(queries, relevant_answers)
        predictions = self.cosine_to_relevance_quantiles(similarity_matrix)

        for metric in self.metrics:
            metric_res = metric.compute(ground_truth, predictions, **kwargs)
            if metric_res is not None:
                self.results.update(metric_res)
            else:
                print("Metric computation fault")

    def get_results(self) -> Dict[str, float]:
        """Get evaluation results

        Returns:
            Dict of metric names and values
        """
        return self.results
