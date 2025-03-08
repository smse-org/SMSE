from smse.logging import get_logger

# from smse.recipes.pipelines.image import (
#     single_image_processing,
#     multiple_images_processing,
#     custom_config_image_processing,
#     center_crop_image_processing,
# )
from smse.recipes.models.imagebind_embedding import imagebind_embedding_example

logger = get_logger(__name__)


def main() -> None:
    imagebind_embedding_example()
