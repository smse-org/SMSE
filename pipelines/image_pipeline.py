from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Callable
from pathlib import Path
import numpy as np
import logging
from enum import Enum
from pipelines.base_pipeline import PipelineConfig, Pipeline


@dataclass
class ImageConfig(PipelineConfig):
    target_size: tuple = (224, 224)
    channels: int = 3
    normalize: bool = True
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


# Image Pipeline
class ImagePipeline(Pipeline):
    def __init__(self, config: ImageConfig):
        super().__init__(config)
        self.config: ImageConfig = config

    def load(self, input_path: Union[str, Path]) -> np.ndarray:
        """Load image from file"""
        try:
            import cv2

            return cv2.imread(str(input_path))
        except ImportError:
            self.logger.warning("OpenCV not found, trying PIL")
            from PIL import Image

            return np.array(Image.open(str(input_path)))

    def validate(self, data: Any) -> bool:
        return isinstance(data, np.ndarray) and len(data.shape) in [2, 3]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image data"""
        import cv2

        # Resize
        image = cv2.resize(image, self.config.target_size)

        # Ensure correct number of channels
        if len(image.shape) == 2:
            image = np.stack([image] * self.config.channels, axis=-1)
        elif image.shape[-1] != self.config.channels:
            if self.config.channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., None]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Normalize
        if self.config.normalize:
            image = image.astype(np.float32) / 255.0
            image = (image - self.config.mean) / self.config.std

        return image
