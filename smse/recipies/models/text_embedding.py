from smse.logging import get_logger
from smse.models import SentenceTransformerTextModel
from smse.types import Modality

logger = get_logger(__name__)


def text_embedding_st_example() -> None:
    """Example of using a SentenceTransformer model."""
    logger.info("=== SentenceTransformer Example ===")

    # Create a SentenceTransformer model
    model = SentenceTransformerTextModel("all-MiniLM-L6-v2")

    # The sentences to encode
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    # Calculate embeddings
    embeddings = model.encode({Modality.TEXT: sentences})[Modality.TEXT]
    logger.info(f"Embeddings shape: {embeddings[0].shape}")

    # Calculate similarity
    similarities = model.similarity(embeddings)
    logger.info(f"Similarities:\n{similarities}")
