from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from smse.pipelines.base import BasePipeline, PipelineConfig

ImageT = NDArray[np.float64 | np.uint8]


@dataclass
class ImageConfig(PipelineConfig):
    """
    Configuration for processing image in a pipeline

    Attributes:
        target_size: Default target size for image resizing.
        color_mode: Default color mode for the image.
        normalizer: Default applicaton of normalization (Do it or not).
        mean: Default mean values for RGB normalization.
        std: Standard deviation values for RGB normalization.
    """

    target_size: tuple[int, int] = (224, 224)
    color_mode: str = "RGB"
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

    def is_bgr(self, image: ImageT) -> bool:
        """Heuristic to determine if an image is likely in BGR Format."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Compare avg intensity of first (blue) and last (red) channels.
            blue_mean = np.mean(image[:, :, 0])
            red_mean = np.mean(image[:, :, 2])
            return bool(blue_mean > red_mean)
        return False

    def default_mode(self, image: ImageT) -> ImageT:
        """Converting image to default mode BGR"""
        import cv2  # type: ignore[import-not-found]

        if not isinstance(image, np.ndarray):  # Assuming PIL Image
            image = np.array(image)

        # Handle different image shapes
        if len(image.shape) == 2:  # Grayscale image
            if self.config.color_mode == "RGBA":
                converted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif self.config.color_mode == "RGB":
                converted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif self.config.color_mode == "BGR":
                converted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                converted_image = image
        elif image.shape[2] == 3:  # RGB or BGR image
            if self.config.color_mode == "RGBA":
                converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            elif self.config.color_mode == "BGR":
                if self.is_bgr(image):
                    converted_image = image  # Already BGR
                else:
                    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif self.config.color_mode == "RGB":
                if self.is_bgr(image):
                    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    converted_image = image  # Already RGB
            else:
                converted_image = image
        elif image.shape[2] == 4:  # RGBA image
            if self.config.color_mode == "RGB":
                converted_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif self.config.color_mode == "BGR":
                converted_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif self.config.color_mode == "BGRA":
                converted_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            else:
                converted_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return ImageT(converted_image)

    def mode_controller(self, image: ImageT, mode: int) -> ImageT:
        import cv2  # type: ignore[import-not-found]

        converted_image = image  # Use a separate variable

        if mode == 0:  # Ensure image mode matches the config color_mode
            converted_image = self.default_mode(image)
        elif mode == 1:  # Convert to gray-scale
            if len(image.shape) == 3 and image.shape[2] == 3:
                if self.is_bgr(image):
                    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif mode == 2:  # Convert to black and white
            if len(image.shape) == 3 and image.shape[2] >= 3:
                if self.is_bgr(image):
                    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                grayscale = image  # Assuming already gray-scale
            _, converted_image = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
        elif mode == 3:  # Convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                if self.is_bgr(image):
                    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:  # Convert from gray-scale
                converted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif mode == 4:  # Convert to BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                if not self.is_bgr(image):
                    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif len(image.shape) == 2:  # Convert from gray-scale
                converted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif mode == 5:  # Convert to RGBA
            if len(image.shape) == 3 and image.shape[2] == 3:
                if self.is_bgr(image):
                    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                else:
                    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            elif len(image.shape) == 2:  # Convert from gray-scale
                converted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        else:
            converted_image = image

        return ImageT(converted_image)

    def preprocess(
        self,
        image: ImageT,
        resizer: Union[bool, tuple[int, int]] = False,
        mode: int = 0,
        median_blur: bool = False,
        normalizer: bool = False,
    ) -> ImageT:
        """Preprocess image data

        Args:
            image (ImageT): Input image, either PIL.image or NumPy array.
            resizer (bool | tuple[int, int]): Resize option; False to skip,
                                                    tuple to specify size.
            mode_converter (int): Mode convertion option:
                0 - Convert to default RGB mode.
                1 - Convert to grayscale.
                2 - Convert to black and white.
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

        # Mode setting, whether to keep it at default if not provided by user
        # or set it as user wishes.
        image = self.mode_controller(image, mode)

        # Noise Reduction
        if median_blur:
            image = cv2.medianBlur(image, 3)  # Applying 3x3 kernel

        # Normalization
        if normalizer and self.config.normalize:
            image = image.astype(np.float32) / 255.0  # Scaling pixel values to [0, 1]
            image = (image - self.config.mean) / self.config.std

        return image
