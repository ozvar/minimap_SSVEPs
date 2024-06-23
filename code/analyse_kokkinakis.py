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
results_dir = root_dir / "results"
fig_dir = root_dir / "results" / "figures"
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
left_minimap_epochs, right_minimap_epochs = utils.load_epochs(data_dir / "kokkinakis" / "epochs.pickle") 
# Otherwise preprocess it
if left_minimap_epochs is None:
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
                    results_dir,
                    random_state=parameters["preprocessing"]["random_state"],
                    max_iter=200
)
rf_metrics = classification.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    RandomForestClassifier,
                    results_dir,
                    random_state=parameters["preprocessing"]["random_state"],
)
svm_metrics = classification.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    SVC,
                    results_dir,
                    random_state=parameters["preprocessing"]["random_state"],
                    kernel="linear"
)


## Here we do spectral analysis on the SSVEP responses
# First compute PSDs and SNRs for each unique combination of parameters
# i.e. each combination of frequency and stimulus side for each condition

# For left minimap condition
left_psds, left_snrs, freqs = spectral_analysis.compute_condition_psds_and_snrs(
    left_minimap_epochs, "left_minimap", parameters)
# For right minimap condition
right_psds, right_snrs, freqs = spectral_analysis.compute_condition_psds_and_snrs(
    right_minimap_epochs, "right_minimap", parameters)

# Now we plot the mean PSDs and SNRs (for all channels)
# For left minimap condition
for psds_key, snrs_key in zip(left_psds, left_snrs):
    vizualisation.plot_PSD_and_SNR(
            psds = left_psds[psds_key],
            snrs = left_snrs[snrs_key],
            freqs = freqs,
            fmin = parameters["analysis"]["fmin"],
            fmax = 25,
            fig_dir = fig_dir,
            fig_suffix = f"{snrs_key}_all_channels"
            )
# For right minimap condition
for psds_key, snrs_key in zip(right_psds, right_snrs):
    vizualisation.plot_PSD_and_SNR(
            psds = right_psds[psds_key],
            snrs = right_snrs[snrs_key],
            freqs = freqs,
            fmin = parameters["analysis"]["fmin"],
            fmax = 25,
            fig_dir = fig_dir,
            fig_suffix = f"{snrs_key}_all_channels" 
            )
