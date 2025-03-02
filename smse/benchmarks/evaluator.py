from smse.benchmarks.metric import Metric
from typing import Dict, List, Any

import numpy as np

class Evaluator:
    """Evaluation framework for models."""

    def __init__(self, model: Any, dataset: Any, metrics: List[Metric]):
        """
        Initialize evaluator

        Args:
            model: Model object that makes predictions
            dataset: Dataset containing features and ground truth
            metrics: List of Metric objects to evaluate
        """
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.results = {}

    def run(self, X: np.ndarray = None, y_target: np.ndarray = None, **kwargs):
        """
        Run evaluation using specificed metrics

        Args:
            X: features to use for prediction
            y_target: ground truth labels
            **kwargs: Additional parametrs for metrics
        """
        if X is None or y_target is None:
            raise ValueError("Features and ground truth must be provided directly.")

        y_pred = self.model.predict(X)
        self.results = {k: v for metric in self.metrics for k, v in metric.compute(y_target, y_pred, **kwargs).items()}

    def get_results(self) -> Dict[str, float]:
        """Get evaluation results

        Returns:
            Dict of metric names and values
        """
        return self.results
