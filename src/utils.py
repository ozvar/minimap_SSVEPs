import logging
import pickle
import os


def load_epochs(data_dir: str):
    with open(os.path.join(data_dir,'epochs.pickle'), 'rb') as f:
        [left_minimap_epochs, right_minimap_epochs] = pickle.load(f)
    return left_minimap_epochs, right_minimap_epochs 


def setup_logger(
        results_dir: str,
        model_type: str,
        ) -> logging.Logger:
    log_dir = os.path.join(results_dir, "logs") 
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{model_type}_classification_metrics.txt")
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

