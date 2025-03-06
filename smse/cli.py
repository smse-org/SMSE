from smse.logging import get_logger
from smse.recipes.pipelines.image import (
    single_image_processing,
    multiple_images_processing,
    image_validation,
    custom_config_image_processing,
)

logger = get_logger(__name__)


def main() -> None:
    logger.info("Running image processing pipeline recipes")
    single_image_processing()
    multiple_images_processing()
    image_validation()
    custom_config_image_processing()
    logger.info("Image processing pipeline recipes completed successfully")
