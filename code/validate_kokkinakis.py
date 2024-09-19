import yaml
import pickle
from pathlib import Path
from src import preprocessing, classification, vizualisation, utils, spectral_analysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from pqdm.processes import pqdm
from typing import List, Dict
import pandas as pd
import pickle

# Declare file locations
# root_dir = Path.home() / "Git" / "minimap_SSVEPs"
root_dir = Path('/home/ozvar/Git/minimap_SSVEPs')
data_dir = root_dir / "data"
logs_dir= root_dir / "results" / "kokkinakis"
fig_dir = root_dir / "results" / "kokkinakis" / "figures"
params_dir = root_dir / "code" / "conf"

# Load params
with open(params_dir / "parameters.yaml", 'r') as file:
    parameters = yaml.safe_load(file)
# Initialize vizualisation settings
vizualisation.sns_styleset()

# Get paths for data files
left_data_folder = preprocessing.get_all_sample_filenames(
    data_dir / "kokkinakis" / "left_minimap_eeg_data",
    "edf"
)
right_data_folder = preprocessing.get_all_sample_filenames(
    data_dir / "kokkinakis" / "right_minimap_eeg_data",
    "edf"
)

# Load data if already preprocessed
try:
    left_minimap_epochs, right_minimap_epochs = utils.load_epochs(data_dir / "kokkinakis" / "epochs.pickle") 
# Otherwise preprocess it
except TypeError:
    left_minimap_epochs = preprocessing.ingest_samples(
            left_data_folder,
            parameters["preprocessing"]
    )
    right_minimap_epochs = preprocessing.ingest_samples(
            right_data_folder,
            parameters["preprocessing"]
    )
    # Equalize the number of epochs for each event between the two sets
    preprocessing.equalize_epoch_counts(
            left_minimap_epochs,
            right_minimap_epochs
    )
    # Pickle the data for later
    with open(data_dir / "kokkinakis" / "epochs.pickle", 'wb') as f:
        pickle.dump([left_minimap_epochs, right_minimap_epochs], f
        )


# Set labels for plots
group_labels = {
    left_minimap_epochs: 'left_group',
    right_minimap_epochs: 'right_group'
}


# ## here we plot time courses of the epochs across conditions
# for group in [left_minimap_epochs, right_minimap_epochs]:
#     for freq in ['3.75', '4.8']:
#         vizualisation.plot_time_course(
#             epochs = group[f'{freq}_LEFT', f'{freq}_RIGHT'],
#             ROIs = parameters['analysis']['vROIs'],
#             group_labels = group_labels,
#             fig_name = f'{group_labels[group]}_{freq}_average_time_courses_visual_channels',
#             fig_dir = fig_dir / "time_courses"
#         )

# ## Here we classify left- versus right- conditioned participants based on their EEG data
# # Cross-validate classifier and return metrics
# log_res_metrics = classification.main_analysis(
#                     left_minimap_epochs,
#                     right_minimap_epochs,
#                     LogisticRegression,
#                     logs_dir,
#                     random_state=parameters["preprocessing"]["random_state"],
#                     permutation_test = False,
#                     max_iter=200
# )
# rf_metrics = classification.main_analysis(
#                     left_minimap_epochs,
#                     right_minimap_epochs,
#                     RandomForestClassifier,
#                     logs_dir,
#                     random_state=parameters["preprocessing"]["random_state"],
#                     permutation_test = False
# )
# svm_metrics = classification.main_analysis(
#                     left_minimap_epochs,
#                     right_minimap_epochs,
#                     SVC,
#                     logs_dir,
#                     random_state=parameters["preprocessing"]["random_state"],
#                     permutation_test = False,
#                     kernel="linear"
# )


# These constants have to be somewhat hardcoded due to the need for multiproc functions to be top-level
## There's a better way to implement this, but let's see if it does what we want it to do first
MODEL_CLASS = RandomForestClassifier
K_FOLDS = 5
RANDOM_STATE = 42

def parallel_lofo(channel_name):
    try:
        if channel_name is not None:
            left_minimap_epochs.drop_channels(channel_name, "warn")
            right_minimap_epochs.drop_channels(channel_name, "warn")

        X_first = classification.compute_psd_and_features(left_minimap_epochs) 
        X_second = classification.compute_psd_and_features(right_minimap_epochs)
        
        print(f"Shape of X_first: {X_first.shape}")
        print(f"Shape of X_second: {X_second.shape}")
        
        X = np.vstack([X_first, X_second])
        print(f"Shape of X after vstack: {X.shape}")
        
        y_first, y_second = classification.create_labels_for_binary_classification(
            len(left_minimap_epochs.events),
            len(right_minimap_epochs.events)
        )
        y = np.concatenate([y_first, y_second])
        print(f"Shape of y: {y}")
        
        print(f"Type of X: {type(X)}, Type of y: {type(y)}")
        
        cv_results = classification.perform_cross_validation(X, y, MODEL_CLASS, K_FOLDS, RANDOM_STATE)
        cv_metrics = classification.parse_cv_results(cv_results, K_FOLDS)
        cv_metrics['dropped_channel'] = channel_name
        return cv_metrics
    except Exception as e:
        print(f"Error in parallel_lofo for channel {channel_name}: {str(e)}")
        return ValueError(str(e))

def prepare_lofo_features(
    first_epochs,
    second_epochs,
):
    """prepare_lofo_features is a helper function which grabs the list of channel names for LOFO importance.
    This also includes an assertion to check that the channel names in both groups match up!

    Args:
        first_epochs (mne.EpochsArray): The first (either left or right) epochs
        second_epochs (mne.EpochsArray): The second (either left or right) epochs
        channel_names (List[str]): The channel names, where each channel is a 'feature' we apply LOFO over

    Returns:
        List[str]: Channel names extracted directly from the epochs array 
    """
    # I think this is a necessary and worthwhile check before running LOFO, since we have two datasets to align.
    assert first_epochs.ch_names == second_epochs.ch_names
    channel_names = first_epochs.ch_names
    channel_names = [None] + channel_names # We include a No feature, as a baseline
    return channel_names
    
def results_to_df(
    results: list
) -> pd.DataFrame:
    parsed = []
    for result in results:
        print(result)
        clean = {
            'dropped_channel' : result['dropped_channel'],
            'accuracy' : result['aggregated_metrics']['accuracy'],
            'precision' : result['aggregated_metrics']['precision'],
            'recall' : result['aggregated_metrics']['recall'],
            'f1' : result['aggregated_metrics']['f1'],
            'fold_scores' : result['fold_scores']
        }
        parsed.append(clean)
    return pd.DataFrame(parsed)

if __name__ == "__main__":

    channel_names = prepare_lofo_features(left_minimap_epochs, right_minimap_epochs)

    # I have chosen to use pqdm to attach a progress bar to this, but using joblib, etc. would be fine too.
    results = pqdm(channel_names[:], parallel_lofo, n_jobs=4)
    results_df = results_to_df(results)

    with open(logs_dir/"rf_validation.pickle", "wb") as results_file:
        pickle.dump(results, results_file)
    results_df.to_csv(logs_dir/"rf_validation.csv")
