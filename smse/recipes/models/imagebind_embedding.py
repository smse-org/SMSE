from smse.logging import get_logger
from smse.models import ImageBindModel
from smse.types import Modality

logger = get_logger(__name__)


def imagebind_embedding_example() -> None:
    """Example of using ImageBind model for image embedding."""
    # Initialize ImageBind model
    model = ImageBindModel()

    # Inputs for different modalities
    images = [
        ".assets/images/bird_image.jpg",
        ".assets/images/car_image.jpg",
        ".assets/images/dog_image.jpg",
    ]

    audio = [
        ".assets/audio/bird_audio.wav",
        ".assets/audio/car_audio.wav",
        ".assets/audio/dog_audio.wav",
    ]

    sentences = [
        "a car",
        "a bird",
        "a dog",
    ]

    inputs = {Modality.IMAGE: images, Modality.TEXT: sentences, Modality.AUDIO: audio}

    # Encode inputs into embeddings
    embeddings = model.encode(inputs)

    logger.info(f"Embedding shape: {embeddings[Modality.TEXT][0].shape}")

    # Print similarities
    img_text_sim = model.cross_modal_similarity(
        embeddings[Modality.IMAGE], embeddings[Modality.TEXT]
    )
    logger.info(f"Image-Text similarity:\n {img_text_sim}")

    img_audio_sim = model.cross_modal_similarity(
        embeddings[Modality.IMAGE], embeddings[Modality.AUDIO]
    )
    logger.info(f"Image-Audio similarity:\n {img_audio_sim}")

    text_audio_sim = model.cross_modal_similarity(
        embeddings[Modality.TEXT], embeddings[Modality.AUDIO]
    )
    logger.info(f"Text-Audio similarity:\n {text_audio_sim}")
