from typing import Any, Dict, Optional

import torch

from smse.logging import get_logger
from smse.models import BaseModel
from smse.types import EmbeddingT, Modality

logger = get_logger(__name__)


class SentenceTransformerTextModel(BaseModel):
    """Wrapper for SentenceTransformer models."""

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        """
        Initialize a SentenceTransformer model.

        Args:
            model_name_or_path: Name or path of a SentenceTransformer model
            device: Device to use for inference (e.g., 'cuda', 'cpu')
        """
        from sentence_transformers import SentenceTransformer

        self.device = (
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = SentenceTransformer(model_name_or_path, device=self.device)
        self._supported_modailities = [Modality.TEXT]

    def encode(self, inputs: Dict[Modality, Any]) -> Dict[Modality, EmbeddingT]:
        """
        Encode text inputs into embeddings.

        Args:
            inputs: Dictionary of text inputs

        Returns:
            Dict[Modality, Tensor]: Dictionary of text embeddings
        """
        embeddings = {}

        for mod, mod_inputs in inputs.items():
            if mod not in self._supported_modailities:
                logger.error(
                    f"Unsupported modality {mod} for {self.__class__.__name__}"
                )

            if mod == Modality.TEXT:
                embeddings[mod] = self.model.encode(mod_inputs, convert_to_tensor=True)

        return embeddings


class SentenceTransformerCLIPModel(SentenceTransformerTextModel):
    """Wrapper for CLIP models."""

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        """
        Initialize a CLIP model.

        Args:
            model_name_or_path: Name or path of a CLIP model
            device: Device to use for inference (e.g., 'cuda', 'cpu')
        """
        super().__init__(model_name_or_path, device=device)
        self._supported_modailities = [Modality.IMAGE, Modality.TEXT]
