from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union


@dataclass
class BaseConfig:
    """Base configuration for all pipelines"""

    batch_size: int = 32
    device: str = "cpu"


class BasePipeline(ABC):
    def __init__(self, config: BaseConfig):
        self.config = config

    @abstractmethod
    def load(self, input_data: Union[str, Path, Any]) -> Any:
        """Load data from file or variable"""
        pass

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Preprocess loaded data"""
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data format and content"""
        pass

    def __call__(self, input_data: Union[Path, Any]) -> Any:
        """Main pipeline execution"""
        if isinstance(input_data, Path):
            data = self.load(input_data)
        else:
            data = input_data

        if not self.validate(data):
            raise ValueError(f"Invalid data format for {self.__class__.__name__}")

        return self.process(data)
