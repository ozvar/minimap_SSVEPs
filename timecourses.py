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

for group in [left_minimap_epochs, right_minimap_epochs]:
    for freq in ['3.75', '4.8']:
        epochs = group[f'{freq}_LEFT', f'{freq}_RIGHT'].pick(ROIs)
        evoked = epochs.average()
        # Create a new figure and subplot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot the average time course
        evoked.plot(
            axes=ax,  # Plot on the specified subplot
            time_unit='s',  # Set the time unit to seconds
            titles=None,  # Remove the default titles
            show=False,  # Do not display the plot yet
            gfp=False,  # Disable Global Field Power (GFP)
            window_title=None,  # Disable the plot window title
            ylim=dict(eeg=[-5, 5]),  # Set the y-axis limits for EEG channels
            spatial_colors=False
        )

        # Customize the plot
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('Average Time Course')
        # Display the plot
        plt.tight_layout()
        plt.savefig(f'{group_labels[group]}_{freq}_average_time_courses_visual_channels.png')
        plt.close()
