import yaml
import os
import pickle
from src import preprocessing, decoding, viz, utils, spectral_analysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Initialize environment 
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
    # Pickle the data for later use
    with open(epochs_path, 'wb') as f:
        pickle.dump([left_minimap_epochs, right_minimap_epochs], f
        )


## Here we plot the time courses of the SSVEP responses
#




"""
## Here we classify left- versus right- conditioned participants based on their EEG data
# Cross-validate classifier and return metrics
log_res_metrics = decoding.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    LogisticRegression,
                    random_state=parameters["RANDOM_STATE"],
                    max_iter=200
)
rf_metrics = decoding.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    RandomForestClassifier,
                    random_state=parameters["RANDOM_STATE"]
)
svm_metrics = decoding.main_analysis(
                    left_minimap_epochs,
                    right_minimap_epochs,
                    SVC,
                    random_state=parameters["RANDOM_STATE"],
                    kernel="linear"
)
"""
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
    viz.plot_PSD_and_SNR(
            psds = left_psds[psds_key],
            snrs = left_snrs[snrs_key],
            freqs = freqs,
            fmin = parameters["analysis"]["fmin"],
            fmax = 25,
            fig_dir = os.path.join("results", "figures"),
            fig_suffix = f"{snrs_key}_all_channels"
            )
# For right minimap condition
for psds_key, snrs_key in zip(right_psds, right_snrs):
    viz.plot_PSD_and_SNR(
            psds = right_psds[psds_key],
            snrs = right_snrs[snrs_key],
            freqs = freqs,
            fmin = parameters["analysis"]["fmin"],
            fmax = 25,
            fig_dir = os.path.join("results", "figures"),
            fig_suffix = f"{snrs_key}_all_channels" 
            )
