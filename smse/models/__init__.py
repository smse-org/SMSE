from smse.logging import get_logger

logger = get_logger(__name__)

try:
    from .sentence_transformers import (
        SentenceTransformerCLIPModel,
        SentenceTransformerTextModel,
    )
except ImportError:
    logger.warning(
        "SentenceTransformer is not installed. "
        "SentenceTransformerModel will not be available.\n"
        "Install with `poetry install --extras sentence-transformers`"
    )

try:
    from .imagebind import ImageBindModel
except ImportError:
    logger.warning(
        "ImageBind is not installed. "
        "ImageBindModel will not be available.\n"
        "Install with `poetry install --extras imagebind`"
    )


__all__ = [
    "ImageBindModel",
    "SentenceTransformerTextModel",
    "SentenceTransformerCLIPModel",
]
