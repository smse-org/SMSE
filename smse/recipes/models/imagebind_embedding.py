from smse.logging import get_logger
from smse.models import ImageBindModel
from smse.pipelines.image import ImageConfig, ImagePipeline
from smse.types import Modality

logger = get_logger(__name__)


def imagebind_embedding_example() -> None:
    """Example of using ImageBind model for image embedding."""
    # Initialize ImageBind model
    model = ImageBindModel()

    # Inputs for different modalities
    image_paths = [
        ".assets/images/bird_image.jpg",
        ".assets/images/car_image.jpg",
        ".assets/images/dog_image.jpg",
    ]

    audio_paths = [
        ".assets/audio/bird_audio.wav",
        ".assets/audio/car_audio.wav",
        ".assets/audio/dog_audio.wav",
    ]

    sentences = [
        "a car",
        "a bird",
        "a dog",
    ]

    image_pipeline = ImagePipeline(
        ImageConfig(
            target_size=(224, 224),
            center_crop=224,
            normalize=True,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
    )

    # Load and process images
    raw_images = [image_pipeline.load(image) for image in image_paths]
    images = image_pipeline.process(raw_images)

    inputs = {
        Modality.IMAGE: images,
        Modality.TEXT: sentences,
        Modality.AUDIO: audio_paths,
    }

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
