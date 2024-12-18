import numpy as np
from typing import Dict, Tuple
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import  cross_validate, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from .utils import logging, setup_logger
from .vizualisation import plot_confusion_matrix 
from mne import concatenate_epochs


def create_labels_for_binary_classification(n_first, n_second):
    """Generate labels for first (0) and second (1) group of participants."""
    labels_first = np.zeros(n_first)
    labels_second = np.ones(n_second)
    return [labels_first, labels_second]


def shuffle_labels_randomly(labels, random_state: int = 42):
    """Shuffle labels randomly."""
    np.random.seed(random_state)
    return np.random.permutation(labels)


def create_labels_for_each_condition(n_left_48, n_left_37, n_right_48, n_right_37):
    """Generate labels for each experimental condition."""
    labels_left_48 = np.zeros(n_left_48)
    labels_left_37 = np.ones(n_left_37)
    labels_right_48 = np.ones(n_right_48) * 2
    labels_right_37 = np.ones(n_right_37) * 3
    return np.concatenate([labels_left_48, labels_left_37, labels_right_48, labels_right_37])


def compute_psd_and_features(
        epochs,
        fmin: float = 0.5,
        fmax: float = 50,
        method='welch'):
    """Compute PSD for epochs and extract features."""
    psd = epochs.compute_psd(fmin=fmin, fmax=fmax, method=method)
    # Extract and average the PSD data to use as features
    X = psd.get_data().mean(axis=-1)
    return X


def compute_band_specific_features(
        epochs,
        bands: Dict[str, Tuple[float, float]]):
    """
    Computes PSD features for specified frequency bands.
    
    Parameters:
    - epochs: Epochs object containing EEG data.
    - bands: Dictionary defining frequency bands, e.g., {'theta': (4, 8), 'alpha': (8, 12), ...}
    
    Returns:
    - A dictionary with band names as keys and feature arrays as values.
    """
    features = {}
    for band, (fmin, fmax) in bands.items():
        psd = epochs.compute_psd(fmin=fmin, fmax=fmax)
        features[band] = psd.get_data().mean(axis=-1)  # Or any other feature extraction method
    return features


def compute_confusion_matrices(cv_results, y_true, X):
    """Compute and average confusion matrices across CV folds.
    
    Parameters:
    -----------
    cv_results : dict
        Results from cross_validate including fitted estimators
    y_true : array
        True labels
    X : array
        Features matrix
        
    Returns:
    --------
    np.ndarray
        Averaged confusion matrix across folds
    """
    confusion_matrices = []
    for estimator in cv_results['estimator']:
        # Get predictions using the full pipeline
        y_pred = estimator.predict(X)
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        confusion_matrices.append(cm)
    # Average the confusion matrices
    avg_cm = np.mean(confusion_matrices, axis=0)
    return avg_cm


def perform_cross_validation(
        X,
        y, 
        model_class,
        k_folds: int,
        random_state: int,
        **model_params):
    """Performs k-fold cross-validation with customizable classifier parameters and returns metrics
    for each fold as well as aggregated scores."""
    pipeline = make_pipeline(StandardScaler(), model_class(**model_params))
    cv = StratifiedKFold(
            n_splits = k_folds,
            shuffle = True,
            random_state = random_state)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score = False,
            return_estimator = True)
    return cv_results


def parse_cv_results(
        cv_results,
        k_folds: int):
    """Extract metrics from each cross-validation fold, computes aggregated scores,
    and parses for readability."""
    # Extract scores for each fold
    fold_scores = {
            f"fold_{i+1}": {metric: scores[i]
                            for metric, scores in cv_results.items()
                            if 'test_' in metric} for i in range(k_folds)}
    # Calculate mean of each metric across CV folds for aggregated scores
    aggregated_metrics = {metric: np.mean(scores)
                          for metric, scores in cv_results.items()
                          if 'test_' in metric}
    # Adjust metric names in aggregated_metrics (remove 'test_' prefix)
    aggregated_metrics = {metric.replace('test_', ''): score
                          for metric, score
                          in aggregated_metrics.items()}
    return {
        "fold_scores": fold_scores,
        "aggregated_metrics": aggregated_metrics
    }

def prepare_clf_data(
    first_epochs,
    second_epochs,
    random_state: int = 42, 
    permutation_test: bool = False,
    method: str = "compute_psd"
): 
    """prepare_clf_data is a helper function designed to generate the usual (X, y) paired data. 
    Note here that it specifically combines the ` first_epochs`  and ` second_epochs`  data in some manner. 
    The default method is the previously used ` compute_psd`  approach, while an alternative concat method is also provided for cases where we do not need cross-val.
    

    Args:
        first_epochs (_type_): The first epoch for combination, usually ` left_minimap_epochs` .
        second_epochs (_type_): Second epoch for combination, usually ` right_minimap_epochs` .
        random_state (int, optional): Seed for rng. Defaults to 42.
        permutation_test (bool, optional): Permutation test if shuffling is required. Defaults to False.
        method (str, optional): String to indicate specified method of combination. Options are ` "compute_psd"`  or ` "concat_epochs"`   Defaults to "compute_psd".

    Returns:
        X (Union[EpochsArray, np.ndarray]): Labelled (X, y) data, where y are the labels generated by ` create_labels_for_binary_classification` .
    """
    # Prepare data for classification
    labels = create_labels_for_binary_classification(
        len(first_epochs.events), 
        len(second_epochs.events)
        )
    if permutation_test:
        labels = shuffle_labels_randomly(labels, random_state)
    if method == "compute_psd":
        X_first = compute_psd_and_features(first_epochs)
        X_second = compute_psd_and_features(second_epochs)
        X = np.vstack([X_first, X_second])
    elif method == "concat_epochs":
        X = concatenate_epochs([first_epochs, second_epochs])
    else:
        raise ValueError(f"Invalid method '{method}'. Expected either 'compute_psd' or 'concat_epochs'.")
    
    return X, labels


def main_analysis(
    first_epochs,
    second_epochs,
    model_class,
    results_dir: Path,
    fig_dir: Path,
    k_folds: int = 5,
    random_state: int = 42,
    permutation_test: bool = False,
    **model_params):
    """
    Main function to perform binary classification of epochs from first and second condition using specified model.
    
    Parameters:
    - first_epochs: Epochs object for the first condition.
    - second_epochs: Epochs object for the second condition.
    - model_class: The classifier model class to be used for analysis.
    - results_dir: Directory for saving log files
    - fig_dir: Directory for saving figures
    - k_folds: Number of CV folds
    - random_state: Random seed for reproducibility.
    - permutation_test: Whether to perform permutation testing
    - model_params: Additional parameters to be passed to the model.
    """
    # Initialize logger
    logger = setup_logger(results_dir, model_class.__name__)
    # Prepare data for classification
    X_first = compute_psd_and_features(first_epochs)
    X_second = compute_psd_and_features(second_epochs)
    Y_first, Y_second = create_labels_for_binary_classification(len(first_epochs.events), len(second_epochs.events))
    X = np.vstack([X_first, X_second])
    y = np.concatenate([Y_first, Y_second])
    np.random.seed(random_state)
    if permutation_test:
        y = shuffle_labels_randomly(y, random_state)
    # Perform cross-validation
    X, y = prepare_clf_data(first_epochs, second_epochs, random_state, permutation_test)
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    # Perform cross-validation on the training set
    logger.info(f"Cross-validating {model_class.__name__}")
    cv_results = perform_cross_validation(X, y, model_class, k_folds, random_state, **model_params)
    cv_metrics = parse_cv_results(cv_results, k_folds)
    # Compute and plot confusion matrix
    avg_cm = compute_confusion_matrices(cv_results, y, X)
    plot_confusion_matrix(
        cm=avg_cm,
        model_name=model_class.__name__,
        fig_dir=fig_dir,
        normalize=True
    )
    # Log results
    logger.info(f"Cross-validation results for {model_class.__name__}:")
    for fold, scores in cv_metrics['fold_scores'].items():
        logger.info(f"Fold {fold}: {scores}")
    logger.info(f"Aggregated scores: {cv_metrics['aggregated_metrics']}")
    # Cleanup
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
