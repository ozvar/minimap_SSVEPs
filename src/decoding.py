import numpy as np
from typing import Dict, Tuple
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import  cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def create_labels_for_left_right_participants(n_left, n_right):
    """Generate labels for left- (0) and right- (1) conditioned participants."""
    labels_left = np.zeros(n_left)
    labels_right = np.ones(n_right)
    return np.concatenate([labels_left, labels_right])


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


def main_analysis(
    left_epochs,
    right_epochs,
    model_class,
    k_folds: int = 5,
    random_state: int = 42,
    permutation_test: bool = False,
    **model_params):
    """
    Main function to perform classification of epochs from left and right conditions using specified model.
    
    Parameters:
    - left_epochs: Epochs object for the left condition.
    - right_epochs: Epochs object for the right condition.
    - model_class: The classifier model class to be used for analysis.
    - test_size: Proportion of the data to be used as the test set.
    - random_state: Random seed for reproducibility.
    - model_params: Additional parameters to be passed to the model.
    """
    labels = create_labels_for_left_right_participants(len(left_epochs.events), len(right_epochs.events))
    if permutation_test:
        labels = shuffle_labels_randomly(labels, random_state)
    X_left = compute_psd_and_features(left_epochs)
    X_right = compute_psd_and_features(right_epochs)
    X = np.vstack([X_left, X_right])
    y = labels
    # Perform cross-validation on the training set
    cv_results = perform_cross_validation(X, y, model_class, k_folds, random_state, **model_params)
    cv_metrics = parse_cv_results(cv_results, k_folds)
    return cv_metrics
