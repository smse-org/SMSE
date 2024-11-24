from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import functional as F  # type: ignore[import-untyped]

from smse.pipelines.base import BasePipeline, PipelineConfig
from smse.types import ImageT


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
        # Initialize the transform Pipeline.
        self.transform_pipeline = self._create_transform_pipeline()

    def _create_transform_pipeline(self) -> transforms.Compose:
        """Create a transformation pipeline for image preprocessing."""
        transform_list = [
            transforms.Lambda(self._convert_mode),  # Ensure specified color mode.
            transforms.ToTensor(),  # Convert PIL to Tensor
        ]

        if self.config.target_size:  # Resize images.
            transform_list.append(transforms.Resize(self.config.target_size))

        if self.config.normalize:  # Normalize images.
            transform_list.append(
                transforms.Normalize(mean=self.config.mean, std=self.config.std)
            )

        return transforms.Compose(transform_list)

    def _convert_mode(self, image: Image.Image) -> Image.Image:
        """
        Convert an image to the specified color mode.

        Args:
            image (ImageT): Input image.

        Returns:
            ImageT: Image converted to the specified mode.
        """
        if self.config.color_mode.upper() == "RGB":
            # Convert to RGB
            image = image.convert("RGB")
        elif self.config.color_mode.upper() == "BGR":
            # Convert to BGR by swapping RGB channels
            image = image.convert("RGB")
            image = Image.fromarray(np.array(image)[..., ::-1])  # Reverse channel order
        elif self.config.color_mode.upper() in ["L", "1", "RGBA", "RGB"]:
            # Convert to the specified mode directly
            image = image.convert(self.config.color_mode.upper())
        else:
            raise ValueError(f"Unsupported target color mode: {self.config.color_mode}")
        return image

    def load(self, input_path: Union[str, Path]) -> ImageT:
        """Load image from file"""
        try:
            import cv2  # type: ignore[import-not-found]

            image: ImageT = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Failed to load image: {input_path}")
            return image
        except ImportError:
            self.logger.warning("OpenCV not found, trying PIL")
            from PIL import Image  # type: ignore[import-not-found]

            return np.array(Image.open(str(input_path)))

    def process(self, images: List[ImageT]) -> torch.Tensor:
        """
        Preprocess a list of images using the transform pipeline.

        Args:
            images (list[PIL.Image | ImageT]): List of input images
            (either PIL or NumPy).

        Returns:
            torch.Tensor: Preprocessed batch of images with shape [B, C, H, W].
        """
        device = torch.device(
            self.config.device
            if torch.cuda.is_available() or self.config.device == "cpu"
            else "cpu"
        )

        processed_images = []
        for image in images:
            if isinstance(image, np.ndarray):  # Convert NumPy to PIL
                image = Image.fromarray(image)
            elif isinstance(image, torch.Tensor):  # Convert tensor to PIL
                image = F.to_pil_image(image)
            processed_images.append(self.transform_pipeline(image))

        return torch.stack(processed_images).to(device)

    def validate(self, data: Any) -> bool:
        return isinstance(data, np.ndarray) and len(data.shape) in [2, 3]
