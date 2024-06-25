import logging
import pickle
import os
from pathlib import Path


def load_epochs(epochs_path: Path):
    try:
        with open(epochs_path, 'rb') as f:
            [left_minimap_epochs, right_minimap_epochs] = pickle.load(f)
        return left_minimap_epochs, right_minimap_epochs 
    except FileNotFoundError:
        print(f"Epochs file not found at {epochs_path}")
        return None


def load_data(data_dir:Path):
    pass


def setup_logger(
        logs_dir: Path,
        model_type: str,
        ) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = logs_dir / f"{model_type}_classification_metrics.txt"
    # Create logger and set the level to INFO
    logger = logging.getLogger("ClassificationLogger")
    logger.setLevel(logging.INFO)
    # Ensure no duplicate handlers
    if not logger.hasHandlers():
        # Create file handler which logs INFO messages
        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
