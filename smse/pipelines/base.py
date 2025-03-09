from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union


@dataclass
class BaseConfig:
    """Base configuration for all pipelines"""

    batch_size: int = 32
    device: str = "cpu"


class BasePipeline(ABC):
    def __init__(self, config: BaseConfig):
        self.config = config

    @abstractmethod
    def load(self, input_data: List[Path]) -> List[Any]:
        """Load data from file or variable"""
        pass

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Preprocess loaded data"""
        pass

    def __call__(self, input_data: Union[List[Path], List[Any]]) -> Any:
        """Main pipeline execution"""
        data: Any
        if all(isinstance(item, Path) for item in input_data):
            data = self.load(input_data)
        else:
            data = input_data

        return self.process(data)
