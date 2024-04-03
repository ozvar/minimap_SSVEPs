import yaml
import os
import pickle
from src import preprocessing, decoding, viz, utils
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Initialize config
parameters_path = os.path.join("conf", "parameters.yaml")
with open(parameters_path, 'r') as file:
    parameters = yaml.safe_load(file)
viz.sns_styleset()

epochs_path = os.path.join("data", "epochs.pickle")

left_data_folder = preprocessing.get_all_sample_filenames(
    "data/left_minimap_eeg_data",
    "edf"
)
right_data_folder = preprocessing.get_all_sample_filenames(
    "data/right_minimap_eeg_data",
    "edf"
)

# Load data if already preprocessed
if os.path.isfile(epochs_path):
    left_minimap_epochs, right_minimap_epochs = utils.load_epochs("data")
# Otherwise preprocess it
else:
    left_minimap_epochs = preprocessing.ingest_samples(
            left_data_folder,
            parameters['preprocessing'])
    right_minimap_epochs = preprocessing.ingest_samples(
            right_data_folder,
            parameters['preprocessing'])
    # Equalize the number of epochs for each event between the two sets
    preprocessing.equalize_epoch_counts(
            left_minimap_epochs,
            right_minimap_epochs)
    # Pickle the data for later use
    with open(epochs_path, "wb") as f:
        pickle.dump([left_minimap_epochs, right_minimap_epochs], f)

# Cross-validate classifier and return metrics
log_res_metrics = decoding.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    LogisticRegression,
                    random_state=parameters['RANDOM_STATE'],
                    max_iter=200)


rf_metrics = decoding.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    RandomForestClassifier,
                    random_state=parameters['RANDOM_STATE'])


svm_metrics = decoding.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    SVC,
                    random_state=parameters['RANDOM_STATE'],
                    kernel='linear')
