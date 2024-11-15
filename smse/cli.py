from smse.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Hello, World1!")
    logger.debug("Hello, World2!")
    logger.warning("Hello, World3!")
    logger.error("Hello, World4!")
    logger.critical("Hello, World5!")
    try:
        raise ValueError("This is a test exception")
    except ValueError:
        logger.exception("An exception occurred")
