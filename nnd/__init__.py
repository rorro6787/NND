from nnd.logger import configure_logging
from pathlib import Path

configure_logging()

CONFIG_NNUNET = Path(__file__).parent / "models" / "nnUNet" / "config" / "dataset.json"
CONFIG_NNUNET_SPLIT = Path(__file__).parent / "models" / "nnUNet" / "config" / "splits_final.json"
