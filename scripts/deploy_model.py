import logging
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.append(PROJECT_ROOT)

from src.common import LOGGER_NAME
from src.hopsworks.client import HopsworksClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

LOGGER = logging.getLogger(LOGGER_NAME)

if __name__ == "__main__":
    # TODO: comment out for production
    LOGGER.setLevel(logging.DEBUG)

    LOGGER.info(f"Getting best model version...")
    best_version = HopsworksClient().get_best_model_version()
    LOGGER.debug(f"Best model version:\n{best_version}")

    LOGGER.info(f"Getting best model...")
    hopsworks_model, _ = HopsworksClient().load_model(version=best_version)
    LOGGER.debug(f"Best model:\n{hopsworks_model}")

    LOGGER.info(f"Deploying model...")
    deployment = HopsworksClient().deploy_model(hopsworks_model, overwrite=True)
    LOGGER.debug(f"Hopsworks Deployment:\n{deployment}")
