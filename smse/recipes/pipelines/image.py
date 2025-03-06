from smse.pipelines.image import ImageConfig, ImagePipeline
from pathlib import Path
import torch
import numpy as np


def single_image_processing() -> None:
    config = ImageConfig()
    pipeline = ImagePipeline(config)

    # Path to a sample image
    image_path = Path(".assets/images/bird_image.jpg")

    # Load and process the image
    image = pipeline.load(image_path)
    processed_image = pipeline.process([image])

    assert isinstance(processed_image, torch.Tensor)
    assert processed_image.shape == (1, 3, config.target_size[0], config.target_size[1])


def multiple_images_processing() -> None:
    config = ImageConfig()
    pipeline = ImagePipeline(config)

    # Paths to sample images
    image_paths = [
        Path(".assets/images/bird_image.jpg"),
        Path(".assets/images/car_image.jpg"),
        Path(".assets/images/dog_image.jpg"),
    ]

    # Load and process the images
    images = [pipeline.load(image_path) for image_path in image_paths]
    processed_images = pipeline.process(images)

    assert isinstance(processed_images, torch.Tensor)
    assert processed_images.shape == (
        3,  # Number of images
        3,
        config.target_size[0],
        config.target_size[1],
    )


def custom_config_image_processing() -> None:
    config = ImageConfig(target_size=(128, 128), color_mode="L", normalize=False)
    pipeline = ImagePipeline(config)

    # Path to a sample image
    image_path = Path(".assets/images/bird_image.jpg")

    # Load and process the image
    image = pipeline.load(image_path)
    processed_image = pipeline.process([image])

    assert isinstance(processed_image, torch.Tensor)
    assert processed_image.shape == (
        1,
        1,
        128,
        128,
    )  # Assuming custom target size and grayscale


def center_crop_image_processing() -> None:
    config = ImageConfig(target_size=(128, 128), center_crop=128)
    pipeline = ImagePipeline(config)

    # Path to a sample image
    image_path = Path(".assets/images/bird_image.jpg")

    # Load and process the image
    image = pipeline.load(image_path)
    processed_image = pipeline.process([image])

    assert isinstance(processed_image, torch.Tensor)
    assert processed_image.shape == (
        1,
        3,
        128,
        128,
    )  # Assuming custom target size and center crop
