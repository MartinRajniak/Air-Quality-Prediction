import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.aqi import load_aqi_data

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# Get a logger instance for your module
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # TODO: comment out for production
    logging.getLogger().setLevel(logging.DEBUG)

    aqi_data = load_aqi_data()
    logger.debug(aqi_data)