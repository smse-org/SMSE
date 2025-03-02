from smse.logging import get_logger
from smse.recipes.models.text_embedding import text_embedding_st_example

logger = get_logger(__name__)


def main() -> None:
    logger.info("Hello Ahmed Saed")
    text_embedding_st_example()
