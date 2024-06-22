import logging
import pickle
import os
from pathlib import Path


def load_epochs(epochs_dir: Path):
    epochs_path = epochs_dir / "epochs.pickle"
    try:
        with open(epochs_dir / "epochs.pickle", 'rb') as f:
            [left_minimap_epochs, right_minimap_epochs] = pickle.load(f)
        return left_minimap_epochs, right_minimap_epochs 
    except FileNotFoundError:
        print(f"Epochs file not found at {epochs_dir}")
        return None


def setup_logger(
        results_dir: Path,
        model_type: str,
        ) -> logging.Logger:
    log_dir = results_dir / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = log_dir / f"{model_type}_classification_metrics.txt"
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
