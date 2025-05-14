from typing import Type

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F  # type: ignore[import-untyped]

from smse.types import EmbeddingT, ImageT


# Image transformations
def transform_image(image: ImageT, to: Type[ImageT]) -> ImageT:
    """Transform image to the specified type: np.ndarray, PIL.Image, or torch.Tensor"""
    if isinstance(image, to):
        return image

    if isinstance(image, np.ndarray):
        if to is Image.Image:
            return F.to_pil_image(image)  # type: ignore[no-any-return]
        elif to is torch.Tensor:
            return torch.from_numpy(image)

    elif isinstance(image, Image.Image):
        if to is np.ndarray:
            return np.array(image)
        elif to is torch.Tensor:
            return torch.from_numpy(np.array(image))

    elif isinstance(image, torch.Tensor):
        if to is np.ndarray:
            return image.numpy()
        elif to is Image.Image:
            return F.to_pil_image(image)  # type: ignore[no-any-return]

    raise TypeError(f"Unsupported conversion from {type(image)} to {to}")


# Embedding transformations
def transform_embedding(embedding: EmbeddingT, to: Type[EmbeddingT]) -> EmbeddingT:
    """Transform embedding to the specified type: List[torch.Tensor], np.ndarray, or torch.Tensor"""
    if isinstance(embedding, to):
        return embedding

    if isinstance(embedding, list):  # List[Tensor] -> other
        if to is torch.Tensor:
            return torch.stack(embedding)
        elif to is np.ndarray:
            return np.array([t.numpy() for t in embedding])

    elif isinstance(embedding, np.ndarray):
        if to is torch.Tensor:
            return torch.tensor(embedding)
        elif to is list:
            return [torch.tensor(arr) for arr in embedding]

    elif isinstance(embedding, torch.Tensor):
        if to is np.ndarray:
            return embedding.numpy()
        elif to is list:
            return list(embedding)

    raise TypeError(f"Unsupported conversion from {type(embedding)} to {to}")
