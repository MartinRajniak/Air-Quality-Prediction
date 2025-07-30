import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.append(PROJECT_ROOT)

from src.common import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

if __name__ == "__main__":
    # TODO: comment out for production
    LOGGER.setLevel(logging.DEBUG)