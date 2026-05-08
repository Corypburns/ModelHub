from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

BASE_PATH = Path.cwd()
DATA_SETS_PATH = BASE_PATH / "DATASETS"
MODEL_BASE_PATH = BASE_PATH / "MODELBASE"
LABEL_MAPS_PATH = BASE_PATH / "LABELMAPS"
