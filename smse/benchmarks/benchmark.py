from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union

import numpy as np
import json

@dataclass
class BenchmarkResult:
    """Base class for storing benchmark results"""
    model_name: str
    dataset_name: str
    task_type: str
    inference_time_ms: float
    samples_per_second: float
    model_size_mb: float = 0
    max_memory_mb: float = 0

    def to_dict(self):
        return asdict(self)

    def save_to_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class BenchmarkTask(ABC):
    """Abstract class for benchmark tasks"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, model, dataset, **kwargs):
        """Run the benchmark task and return results"""
        pass

    @abstractmethod
    def get_metrics(self):
        """Return the list of metrics this task measures"""
        pass

class ModelBenchmark(ABC):
    """Abstract class for model benchmarks"""

    def __init__(self, name: str, tasks: List[BenchmarkTask]) -> None:
        self.name = name
        self.tasks = tasks
        self.results = {}

    @abstractmethod
    def load_model(self, model_name, **kwargs):
        """Load a model by name"""
        pass

    @abstractmethod
    def load_dataset(self, dataset_name, **kwargs):
        """Load a dataset by name"""
        pass

    def store_result(self, task_name, model_name, dataset_name, result):
        """Log a benchmark result"""
        if task_name not in self.results:
            self.results[task_name] = {}
        if model_name not in self.results[task_name]:
            self.results[task_name][model_name] = {}

        self.results[task_name][model_name][dataset_name] = result

    def run_benchmark(self, model_names, dataset_names, **kwargs):
        """Run all benchmark tasks for all specified models and datasets"""
        for model_name in model_names:
            model = self.load_model(model_name)

            for dataset_name in dataset_names:
                dataset = self.load_dataset(dataset_name)

                for task in self.tasks:
                    print(f"Running task {task.name} for model {model_name} on data-set {dataset_name}")
                    result = task.run(model, dataset, **kwargs)
                    self.store_result(task.name, model_name, dataset_name, result)

        return self.results

    def _get_primary_metric_for_task(self, task_name):
        """Helper to determine the primary metric for a task"""
        for task in self.tasks:
            if task.name == task_name:
                metrics = task.get_metrics()
                if metrics:
                    return metrics[0]
        return "score"

    def _extract_metric_values(self, result, metric_name):
        """Extract specific metric value f"""