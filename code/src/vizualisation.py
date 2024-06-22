import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# configure pandas table display
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def sns_styleset():
    """Configure parameters for plotting"""
    sns.set_theme(context='paper',
                  style='whitegrid',
                  # palette='deep',
                  palette=['#c44e52',
                           '#8c8c8c',
                           '#937860',
                           '#ccb974',
                           '#4c72b0',
                           '#dd8452']
                  )
    mpl.rcParams['figure.dpi']        = 300
    mpl.rcParams['axes.linewidth']    = 1
    mpl.rcParams['grid.color']        = '.8'
    mpl.rcParams['axes.edgecolor']    = '.15'
    mpl.rcParams['axes.spines.right']        = False
    mpl.rcParams['axes.spines.top']          = False
    mpl.rcParams['xtick.bottom']      = True
    mpl.rcParams['ytick.left']        = True
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['ytick.major.width'] = 1
    mpl.rcParams['xtick.color']       = '.15'
    mpl.rcParams['ytick.color']       = '.15'
    mpl.rcParams['xtick.major.size']  = 3
    mpl.rcParams['ytick.major.size']  = 3
    mpl.rcParams['font.size']         = 14
    mpl.rcParams['axes.titlesize']    = 14
    mpl.rcParams['axes.labelsize']    = 13
    mpl.rcParams['legend.fontsize']   = 14
    mpl.rcParams['legend.frameon']    = False
    mpl.rcParams['xtick.labelsize']   = 13
    mpl.rcParams['ytick.labelsize']   = 13


def plot_PSD_and_SNR(
    psds: np.ndarray,
    snrs: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    fig_dir: str,
    fig_suffix: str
):
    # Setup figure and axes
    fig, axes = plt.subplots(2,
                             1,
                             sharex = True,
                             figsize = (8, 5))
    freq_range = range(
        np.where(freqs >= fmin)[0][0],
        np.where(freqs <= fmax)[0][-1] + 1  # +1 to include the upper bound
    )
    # Compute mean and standard deviation of PSDs 
    psds_plot = 10 * np.log10(psds)
    psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
    psds_std = psds_plot.std(axis=(0, 1))[freq_range]
    # Plot PSDs
    palette = sns.color_palette()
    axes[0].plot(freqs[freq_range], psds_mean, color=palette[1])
    axes[0].fill_between(
        freqs[freq_range],
        psds_mean - psds_std,
        psds_mean + psds_std,
        alpha=0.2,
        color=palette[1]
    ) 
    axes[0].set(ylabel="PSD [dB]")
    # Compute mean and standard deviation of SNRs 
    snrs_mean = snrs.mean(axis=(0, 1))[freq_range]
    snrs_std = snrs.std(axis=(0, 1))[freq_range]
    # Plot SNRs
    axes[1].plot(freqs[freq_range], snrs_mean, color=palette[0])
    axes[1].fill_between(
        freqs[freq_range],
        snrs_mean - snrs_std,
        snrs_mean + snrs_std,
        alpha = 0.2,
        color=palette[0]
    )
    axes[1].set(xlabel = "Frequency [Hz]",
                ylabel = "SNR [dB]",
                ylim = [-2, 10],
                xlim = [fmin, fmax]
    )
    fig.align_labels()
    #plt.suptitle(f"PSD and SNR for {fig_suffix}")
    # Save figures in raster and vector formats
    fig_path = os.path.join(
            fig_dir,
            f"PSD_and_SNR_{fig_suffix}"
            )
    plt.savefig(f'{fig_path}.png', bbox_inches='tight')
    plt.savefig(f'{fig_path}.svg', bbox_inches='tight')
    plt.close()


def plot_fft_magnitudes(
    fft_mags_mean: np.ndarray,
    fft_mags_std: np.ndarray,
    freqs: np.ndarray,
    group: str,
    condition: str,
    ROIs: str,
    fig_dir: str,
    color: tuple
):
    """Plot magnitudes of the FFT of epochs against frequencies of interest."""
    plt.plot(
        freqs,
        fft_mags_mean,
        color = color
    )
    plt.fill_between(
        freqs,
        fft_mags_mean - fft_mags_std,
        fft_mags_mean + fft_mags_std,
        alpha = 0.3,
        color = color 
    )
    plt.ylim(bottom = 0)
    plt.xlim(left = 1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT magnitude')
    # Name and save the plot
    fig_name = f'mean_FFT_magnitudes_{group}_{condition}_{ROIs}.png'
    fig_dir = os.path.join(fig_dir, 'FFT_magnitudes')
    os.makedirs(fig_dir, exist_ok = True)
    plt.savefig(os.path.join(fig_dir, fig_name))
    plt.close()


def plot_snr(
    snr_mean: np.ndarray,
    freqs: np.ndarray,
    group: str,
    condition: str,
    ROIs: str,
    fig_dir: str,
    color: tuple
):
    """Plot SNR of the FFT of epochs against frequencies of interest."""
    plt.plot(
        freqs,
        snr_mean,
        color = color
    )
    plt.ylim(bottom = 1)
    plt.xlim(left = 1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SNR')
    plt.title(f'{group.replace("_", " ").title()} {condition.title()} {ROIs.replace("_", " ").title()}')
    # Name and save the plot
    fig_name = f'mean_SNR_{group}_{condition}_{ROIs}.png'
    fig_dir = os.path.join(fig_dir, 'SNR')
    os.makedirs(fig_dir, exist_ok = True)
    plt.savefig(os.path.join(fig_dir, fig_name), bbox_inches = 'tight')
    plt.close()
