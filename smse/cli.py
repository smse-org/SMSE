from smse.logging import get_logger
from smse.recipes.benchmark.text_evaluation import text_evaluator_st_example

logger = get_logger(__name__)


def main() -> None:
    logger.info("Hello Ahmed Saed")
    text_evaluator_st_example()
