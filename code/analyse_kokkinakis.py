import yaml
import pickle
from pathlib import Path
from src import preprocessing, classification, vizualisation, utils, spectral_analysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Declare file locations
root_dir = Path.home() / "Git" / "minimap_SSVEPs"
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


## Here we plot time courses of the epochs across conditions
for group in [left_minimap_epochs, right_minimap_epochs]:
    for freq in ['3.75', '4.8']:
        vizualisation.plot_time_course(
            epochs = group[f'{freq}_LEFT', f'{freq}_RIGHT'],
            ROIs = parameters['analysis']['vROIs'],
            group_labels = group_labels,
            fig_name = f'{group_labels[group]}_{freq}_average_time_courses_visual_channels',
            fig_dir = fig_dir / "time_courses"
        )

## Here we classify left- versus right- conditioned participants based on their EEG data
# Cross-validate classifier and return metrics
log_res_metrics = classification.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    LogisticRegression,
                    logs_dir,
                    random_state=parameters["preprocessing"]["random_state"],
                    permutation_test = False,
                    max_iter=200
)
rf_metrics = classification.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    RandomForestClassifier,
                    logs_dir,
                    random_state=parameters["preprocessing"]["random_state"],
                    permutation_test = False
)
svm_metrics = classification.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    SVC,
                    logs_dir,
                    random_state=parameters["preprocessing"]["random_state"],
                    permutation_test = False,
                    kernel="linear"
)
