from pathlib import Path

from imagebind.data import return_bpe_path  # type: ignore[import]
from imagebind.models.multimodal_preprocessors import (  # type: ignore[import]
    SimpleTokenizer,
)

from smse.logging import get_logger
from smse.models import ImageBindModel
from smse.pipelines.image import ImageConfig, ImagePipeline
from smse.pipelines.text import TextConfig, TextPipeline
from smse.pipelines.audio import AudioConfig, AudioPipeline
from smse.types import Modality
from smse.device import get_device

logger = get_logger(__name__)


def imagebind_embedding_example() -> None:
    """Example of using ImageBind model for image embedding."""
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize ImageBind model
    model = ImageBindModel(device=device)

    # Inputs for different modalities
    image_paths = [
        Path(".assets/images/bird_image.jpg"),
        Path(".assets/images/car_image.jpg"),
        Path(".assets/images/dog_image.jpg"),
    ]

    audio_paths = [
        Path(".assets/audio/bird_audio.wav"),
        Path(".assets/audio/car_audio.wav"),
        Path(".assets/audio/dog_audio.wav"),
    ]

    sentences = [
        "a bird",
        "a car",
        "a dog",
    ]

    # Load and process images
    image_pipeline = ImagePipeline(
        ImageConfig(
            target_size=(224, 224),
            center_crop=224,
            normalize=True,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
            device=device,
        )
    )

    processed_images = image_pipeline(image_paths)

    # Load and process text
    text_pipeline = TextPipeline(
        TextConfig(
            chunk_size=240,
            chunk_overlap=10,
            tokenizer=SimpleTokenizer(bpe_path=return_bpe_path()),
            device=device,
        )
    )

    processed_text = text_pipeline.process(sentences)

    # Load and process audio with our custom pipeline
    audio_config = AudioConfig(
        sampling_rate=16000,
        mono=True,
        normalize_audio=False,
        use_clips=True,
        apply_melspec=True,
        num_mel_bins=128,
        target_length=204,
        clip_duration=2.0,
        clips_per_audio=3,
        mean=-4.268,
        std=9.138,
        device=device,
    )

    audio_pipeline = AudioPipeline(audio_config)

    # Process audio files - pipeline now handles everything in one step
    # and returns a tensor directly in the format expected by ImageBind
    audio_tensor = audio_pipeline(audio_paths)

    inputs = {
        Modality.TEXT: processed_text,
        Modality.IMAGE: processed_images,
        Modality.AUDIO: audio_tensor,
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

    audio_text_sim = model.cross_modal_similarity(
        embeddings[Modality.AUDIO], embeddings[Modality.TEXT]
    )
    logger.info(f"Audio-Text similarity:\n {audio_text_sim}")
