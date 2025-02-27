from smse.logging import get_logger
from smse.models import SentenceTransformerCLIPModel
from smse.types import Modality

logger = get_logger(__name__)


def image_embedding_st_clip_example() -> None:
    """Example of using a SentenceTransformer model."""
    logger.info("=== SentenceTransformer CLIP Example ===")

    # Create a SentenceTransformer model
    model = SentenceTransformerCLIPModel("clip-ViT-L-14")

    # The images to encode
    images = [
        "https://raw.githubusercontent.com/facebookresearch/ImageBind/refs/heads/main/.assets/bird_image.jpg",
        "https://raw.githubusercontent.com/facebookresearch/ImageBind/refs/heads/main/.assets/car_image.jpg",
        "https://raw.githubusercontent.com/facebookresearch/ImageBind/refs/heads/main/.assets/dog_image.jpg",
    ]

    sentences = [
        "a car",
        "a bird",
        "a dog",
    ]

    # Calculate embeddings
    embeddings = model.encode({Modality.IMAGE: images, Modality.TEXT: sentences})
    logger.info(f"Embeddings shape: {embeddings[Modality.TEXT][0].shape}")

    # Calculate similarity
    similarities = model.similarity(
        embeddings[Modality.TEXT], embeddings[Modality.IMAGE]
    )
    logger.info(f"Similarities:\n{similarities}")
