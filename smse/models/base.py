import abc
from typing import Any, Dict, List, Optional

import torch
from numpy import ndarray
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
        self, embeddings1: EmbeddingT, embeddings2: Optional[EmbeddingT] = None
    ) -> Tensor:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings. If None, calculate similarity within embeddings1

        Returns:
            Tensor: Cosine similarity matrix
        """

        def to_tensor(embeddings: EmbeddingT) -> Tensor:
            if isinstance(embeddings, list):
                embeddings = torch.cat(embeddings, dim=0)
            elif isinstance(embeddings, ndarray):
                embeddings = torch.from_numpy(embeddings)
            return embeddings

        embeddings1 = to_tensor(embeddings1)
        embeddings2 = to_tensor(embeddings2) if embeddings2 is not None else embeddings1

        # Normalize embeddings
        embeddings1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)

        # Check if embeddings are tensors
        assert isinstance(embeddings1, Tensor)
        assert isinstance(embeddings2, Tensor)

        # Calculate cosine similarity
        return torch.mm(embeddings1, embeddings2.T)
