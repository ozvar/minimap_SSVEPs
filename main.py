import yaml
import os
import mne
from src import preprocessing, decoding, viz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Initialize config
parameters_path = os.path.join("conf", "parameters.yaml")
with open(parameters_path, 'r') as file:
    parameters = yaml.safe_load(file)
viz.sns_styleset()

left_data_folder = preprocessing.get_all_sample_filenames(
    "data/left_minimap_eeg_data",
    "edf"
)
right_data_folder = preprocessing.get_all_sample_filenames(
    "data/right_minimap_eeg_data",
    "edf"
)

# Preprocess data
left_minimap_epochs = preprocessing.ingest_samples(left_data_folder)
right_minimap_epochs = preprocessing.ingest_samples(right_data_folder)

left_minimap_epochs = mne.concatenate_epochs(left_minimap_epochs)
right_minimap_epochs = mne.concatenate_epochs(right_minimap_epochs)

# Equalize the number of epochs for each event between the two sets
preprocessing.equalize_epoch_counts(left_minimap_epochs, right_minimap_epochs)

# Cross-validate classifier and return metrics
metrics = decoding.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    RandomForestClassifier,
                    random_state=parameters['RANDOM_STATE'])
