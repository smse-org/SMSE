from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Any
from pathlib import Path
import logging
from enum import Enum


# Base configurations and types
class DataType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class PipelineConfig:
    """Base configuration for all pipelines"""

    input_type: DataType
    max_sequence_length: Optional[int] = None
    batch_size: int = 32
    device: str = "cpu"


class Pipeline(ABC):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load(self, input_data: Union[str, Path, Any]) -> Any:
        """Load data from file or variable"""
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess loaded data"""
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data format and content"""
        pass

    def __call__(self, input_data: Union[str, Path, Any]) -> Any:
        """Main pipeline execution"""
        if isinstance(input_data, (str, Path)):
            data = self.load(input_data)
        else:
            data = input_data

        if not self.validate(data):
            raise ValueError(f"Invalid data format for {self.__class__.__name__}")

        return self.preprocess(data)
