import abc
from ast import Dict
from typing import Any, List, Optional

import torch
from torch import Tensor

from smse.types import EmbeddingT, Modality


class BaseModel(abc.ABC):
    """Abstract base class for all models in SMSE framework."""

    _supported_modailities: List[Modality] = []

    @abc.abstractmethod
    def encode(self, inputs: Dict[Modality, Any]) -> Dict[Modality, EmbeddingT]:
        """
        Encode inputs of a specific modality into embeddings.

        Args:
            inputs: Input data to encode
            modality: The modality of the input data

        Returns:
            Tensor: Embeddings of the input data
        """
        pass

    def get_supported_modalities(self) -> List[Modality]:
        """
        Get list of modalities supported by this model.

        Returns:
            List[Modality]: List of supported modalities
        """
        return self._supported_modailities

    def similarity(
        self, embeddings1: Tensor, embeddings2: Optional[Tensor] = None
    ) -> Tensor:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings. If None, calculate similarity within embeddings1

        Returns:
            Tensor: Cosine similarity matrix
        """
        if embeddings2 is None:
            embeddings2 = embeddings1

        # Normalize embeddings
        embeddings1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)

        # Calculate cosine similarity
        return torch.mm(embeddings1, embeddings2.T)
