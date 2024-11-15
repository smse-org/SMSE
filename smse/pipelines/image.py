from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from smse.pipelines.base import Pipeline, PipelineConfig

ImageT = NDArray[np.float64 | np.uint8]


@dataclass
class ImageConfig(PipelineConfig):
    target_size: tuple[int, int] = (224, 224)
    channels: int = 3
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


# Image Pipeline
class ImagePipeline(Pipeline):
    def __init__(self, config: ImageConfig):
        super().__init__(config)
        self.config: ImageConfig = config

    def load(self, input_path: Union[str, Path]) -> ImageT:
        """Load image from file"""
        try:
            import cv2  # type: ignore[import-not-found]

            image: ImageT = cv2.imread(str(input_path))

            return image
        except ImportError:
            self.logger.warning("OpenCV not found, trying PIL")
            from PIL import Image  # type: ignore[import-not-found]

            return np.array(Image.open(str(input_path)))

    def validate(self, data: Any) -> bool:
        return isinstance(data, np.ndarray) and len(data.shape) in [2, 3]

    def preprocess(self, image: ImageT) -> ImageT:
        """Preprocess image data"""
        import cv2  # type: ignore[import-not-found]

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
