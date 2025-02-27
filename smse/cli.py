from smse.logging import get_logger
from smse.recipies.models.st_models import sentence_transformer_example

logger = get_logger(__name__)


def main() -> None:
    sentence_transformer_example()
