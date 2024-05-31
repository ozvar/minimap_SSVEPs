import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import viz, utils
from src.spectral_analysis import snr_spectrum, compute_fft_mag, compute_snr_for_epochs


# Initialize environment 
parameters_path = os.path.join("conf", "parameters.yaml")
with open(parameters_path, 'r') as file:
    parameters = yaml.safe_load(file)
viz.sns_styleset()

COLOR = sns.color_palette()[1]
FIG_DIR = os.path.join('results', 'figures')
ROIs = parameters['analysis']['vROIs']

# Load data if already preprocessed
epochs_path = os.path.join("data", "epochs.pickle")
if os.path.isfile(epochs_path):
    left_minimap_epochs, right_minimap_epochs = utils.load_epochs("data")

group_labels = {
    left_minimap_epochs: 'left_group',
    right_minimap_epochs: 'right_group'
}
### DO ANALYSIS
for group in [left_minimap_epochs, right_minimap_epochs]:
    for condition in ['3.75_LEFT', '3.75_RIGHT', '4.8_LEFT', '4.8_RIGHT']:
        # get the epochs for each group and condition
        epochs = group[condition]
        # compute magnitudes of the fft and get frequencies
        fft_mags, freqs = compute_fft_mag(
            epochs = epochs,
            analysis_params = parameters['analysis']
        )
        snr, freqs = compute_snr_for_epochs(
            epochs = epochs,
            analysis_params = parameters['analysis']
        )
        fft_mags_mean = np.nanmean(fft_mags, axis=(0, 1))
        fft_mags_std = np.nanstd(fft_mags, axis=(0, 1))
        snr_mean = np.nanmean(snr, axis=(0, 1))
        # Plot fft magnitudes against the desired frequencies
        viz.plot_fft_magnitudes(
            fft_mags_mean,
            fft_mags_std,
            freqs,
            group_labels[group],
            condition,
            ROIs='all_channels',
            fig_dir=FIG_DIR,
            color=sns.color_palette()[1]
        )
        # Now plot SNR against the desired frequencies
        viz.plot_snr(
            snr_mean,
            freqs,
            group_labels[group],
            condition,
            ROIs='all_channels',
            fig_dir=FIG_DIR,
            color=sns.color_palette()[1]
        )
### REPEAT FOR VISUAL CHANNELS
for group in [left_minimap_epochs, right_minimap_epochs]:
    for condition in ['3.75_LEFT', '3.75_RIGHT', '4.8_LEFT', '4.8_RIGHT']:
        # get the epochs for each group and condition
        epochs = group[condition].pick(ROIs)
        # compute magnitudes of the fft and get frequencies
        fft_mags, freqs = compute_fft_mag(
            epochs = epochs,
            analysis_params = parameters['analysis']
        )
        snr, freqs = compute_snr_for_epochs(
            epochs = epochs,
            analysis_params = parameters['analysis']
        )
        fft_mags_mean = np.nanmean(fft_mags, axis=(0, 1))
        fft_mags_std = np.nanstd(fft_mags, axis=(0, 1))
        snr_mean = np.nanmean(snr, axis=(0, 1))
        # Plot fft magnitudes against the desired frequencies
        viz.plot_fft_magnitudes(
            fft_mags_mean,
            fft_mags_std,
            freqs,
            group_labels[group],
            condition,
            ROIs='visual_channels',
            fig_dir=FIG_DIR,
            color=sns.color_palette()[1]
        )
        # Now plot SNR against the desired frequencies
        viz.plot_snr(
            snr_mean,
            freqs,
            group_labels[group],
            condition,
            ROIs='visual_channels',
            fig_dir=FIG_DIR,
            color=sns.color_palette()[1]
        )
