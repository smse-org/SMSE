from typing import Any, Dict, Optional

import torch
from numpy import ndarray
from torch import Tensor

from smse.device import get_device
from smse.logging import get_logger
from smse.models.base import BaseModel
from smse.types import EmbeddingT, Modality

logger = get_logger(__name__)


class ImageBindModel(BaseModel):
    """Wrapper for ImageBind models."""

    def __init__(self, pretrained: bool = True, device: Optional[str] = None):
        """
        Initialize an ImageBind model.

        Args:
            pretrained: Whether to load pretrained weights
            device: Device to use for inference (e.g., 'cuda', 'cpu')
        """
        from imagebind.models import imagebind_model  # type: ignore[import]
        from imagebind.models.imagebind_model import (  # type: ignore[import]
            ModalityType,
        )

        self.device = device if device is not None else get_device()
        self.model = imagebind_model.imagebind_huge(pretrained=pretrained)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Mapping from SMSE modalities to ImageBind modalities
        self.modality_mapping = {
            Modality.TEXT: ModalityType.TEXT,
            Modality.IMAGE: ModalityType.VISION,
            Modality.AUDIO: ModalityType.AUDIO,
        }
        self._supported_modailities = list(self.modality_mapping.keys())

    def encode(self, inputs: Dict[Modality, Any]) -> Dict[Modality, EmbeddingT]:
        """
        Encode inputs from different modalities into embeddings.

        Args:
            inputs: Dictionary of inputs for different modalities
                    For example: {Modality.TEXT: ["dog", "cat"], Modality.IMAGE: ["img1.jpg", "img2.jpg"]}
            modality: If specified, return embeddings only for this modality

        Returns:
            Dict[Modality, Tensor] or Tensor: Dictionary of embeddings for each modality or tensor for specific modality
        """
        # Convert inputs to ImageBind format
        imagebind_inputs = {}

        for mod, mod_inputs in inputs.items():
            if mod not in self._supported_modailities:
                logger.error(
                    f"Unsupported modality {mod} for {self.__class__.__name__}"
                )

            imagebind_mod = self.modality_mapping[mod]

            if mod == Modality.TEXT:
                imagebind_inputs[imagebind_mod] = mod_inputs
            elif mod == Modality.IMAGE:
                imagebind_inputs[imagebind_mod] = mod_inputs
            elif mod == Modality.AUDIO:
                imagebind_inputs[imagebind_mod] = mod_inputs

        # Generate embeddings
        with torch.no_grad():
            imagebind_embeddings = self.model(imagebind_inputs)

        # Convert back to SMSE format
        embeddings = {
            mod: imagebind_embeddings[self.modality_mapping[mod]]
            for mod in inputs.keys()
        }

        return embeddings

    def cross_modal_similarity(
        self,
        embeddings1: EmbeddingT,
        embeddings2: Optional[EmbeddingT] = None,
    ) -> Tensor:
        """
        Calculate cross-modal similarity between embeddings from different modalities.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Tensor: Softmax normalized similarity matrix
        """

        def to_tensor(embeddings: EmbeddingT) -> Tensor:
            if isinstance(embeddings, list):
                embeddings = torch.cat(embeddings, dim=0)
            elif isinstance(embeddings, ndarray):
                embeddings = torch.from_numpy(embeddings)
            return embeddings

        embeddings1 = to_tensor(embeddings1)
        embeddings2 = to_tensor(embeddings2) if embeddings2 is not None else embeddings1

        similarity = torch.softmax(embeddings1 @ embeddings2.T, dim=-1)
        return similarity
