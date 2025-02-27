from smse.logging import get_logger
from smse.recipies.models.imagebind_embedding import imagebind_embedding_example

logger = get_logger(__name__)


def main() -> None:
    imagebind_embedding_example()
