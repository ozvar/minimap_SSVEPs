import yaml
import os
import mne
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
    for condition_freq in [3.75, 4.8]:
        for direction in ['LEFT', 'RIGHT']:
            condition = f'{str(condition_freq)}_{direction}'
            print(condition)
            # get the epochs for each group and condition
            epochs = group[condition]
            # compute magnitudes of the fft and get frequencies
            snr, freqs = compute_snr_for_epochs(
                epochs = epochs,
                analysis_params = parameters['analysis']
            )
            for i, freq in enumerate(freqs):
                data = snr.mean(axis=(0))[:62, i] # we drop the last (trigger) channel
                mne.viz.plot_topomap(data, epochs.info)
                fig_name = f'{group_labels[group]}_{condition}_{str(np.round(freq, 2))}Hz_topomap.png'
                fig_dir = os.path.join(FIG_DIR, 'topomaps')
                os.makedirs(fig_dir, exist_ok = True)
                plt.savefig(os.path.join(fig_dir, fig_name), bbox_inches = 'tight')
                plt.close()
