from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from smse.pipelines.base import BasePipeline, PipelineConfig

ImageT = NDArray[np.float64 | np.uint8]


@dataclass
class ImageConfig(PipelineConfig):
    target_size: tuple[int, int] = (224, 224)
    channels: int = 3
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


# Image Pipeline
class ImagePipeline(BasePipeline):
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

    def default_mode(self, image: ImageT) -> ImageT:
        """Converting image to default mode BGR"""
        import cv2  # type: ignore[import-not-found]

        if not isinstance(image, np.ndarray):  # Assuming PIL Image
            image = np.array(image)

        if len(image.shape) == 2:  # Grayscale
            converted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:  # RGB or BGR, ensure BGR.
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:  # RGBA
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return converted_image

    def preprocess(
        self,
        image: ImageT,
        resizer: Union[bool, tuple[int, int]] = False,
        mode_converter: int = 0,
        median_blur: bool = False,
        normalizer: bool = False,
    ) -> ImageT:
        """Preprocess image data

        Args:
            image (ImageT): Input image, either PIL.image or NumPy array.
            resizer (bool | tuple[int, int]): Resize option; False to skip,
                                                    tuple to specify size.
            mode_converter (int): Mode convertion option:
                0 - Convert to default BGR mode
                1 - Convert to grayscale.
                2 - Convert to black and white
            median_blur (bool): Whether to apply median blur for noise reduction or not.
            normalizer (bool): Whether to normalize the image or not.

        Returns:
            ImageT: Preprocessed image as a NumPy array.
        """
        import cv2  # type: ignore[import-not-found]

        # Ensure image is a NumPy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Resizing
        if resizer:
            if isinstance(resizer, tuple):
                target_size = resizer
            else:
                target_size = self.config.target_size
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # Color Conversion
        if mode_converter == 0:  # Ensure default image mode is BGR
            image = self.default_mode(image)
        elif mode_converter == 1:  # Convert to gray-scale
            image = cv2.cvtColor(
                image,
            )
        elif mode_converter == 2:  # Convert to black and white
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, image = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)

        # Noise Reduction
        if median_blur:
            image = cv2.medianBlur(image, 3)  # Applying 3x3 kernel

        # Normalization
        if normalizer and self.config.normalize:
            image = image.astype(np.float32) / 255.0  # Scaling pixel values to [0, 1]
            image = (image - self.config.mean) / self.config.std

        return image
