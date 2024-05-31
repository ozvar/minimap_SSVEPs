import mne
import numpy as np
from typing import List, Optional


def compute_psd(
    epochs,
    preprocessing_params: dict,
    analysis_params: dict,
    vROI: Optional[List[str]] = None,
    verbose: bool = False
):
    """Compute the power spectral density (PSD) of the epochs."""
    sfreq = epochs.info["sfreq"]
    n_fft = int(sfreq * (preprocessing_params["tmax"] - preprocessing_params["tmin"]))
    spectrum = epochs.compute_psd(
            method = "welch",
            n_fft = n_fft,
            n_overlap = 0,
            n_per_seg = None,
            fmin = analysis_params["fmin"],
            fmax = analysis_params["fmax"],
            window = "boxcar",
            verbose = verbose
    )
    if vROI:
        spectrum.pick(vROI)
    psds, freqs = spectrum.get_data(return_freqs=True)
    return psds, freqs


def compute_fft_mag(
    epochs,
    analysis_params: dict
):
    """Compute response magnitude of fourier transformed epochs"""
    data = epochs.get_data()
    # Compute FFT for each epoch and channel
    fft_output = np.fft.fft(data, axis=-1)
    fft_mag = np.abs(fft_output)
    # Slice the frequencies within our desired range
    freqs = np.fft.fftfreq(data.shape[-1], 1/epochs.info['sfreq'])
    freqs = freqs[
            (freqs >= analysis_params['fmin']) &
            (freqs <= analysis_params['fmax'])
            ]
    # Slice magnitudes up to the desired frequency
    fft_mag = fft_mag[:, :, :len(freqs)]

    return fft_mag, freqs


def snr_spectrum(
    fft,
    noise_n_neighbor_freqs: int,
    noise_skip_neighbor_freqs: int
):
    """Compute SNR spectrum from frequency spectrum using convolution.

    Parameters
    ----------
    fft : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing magnitude values obtained after FFT. Also works
        with arrays as produced by MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()
    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda fft_: np.convolve(fft_, averaging_kernel, mode="valid"), axis=-1, arr=fft
    )
    # The mean is not defined on the edges so we will pad it with nans. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)
    snr = fft / mean_noise
    snr = np.nan_to_num(snr, nan=0, posinf=0, neginf=0)
    
    return snr


def compute_snr_for_epochs(
    epochs,
    analysis_params: dict,
):
    # Compute magnitudes of the FFT across epochs 
    fft_mag, freqs = compute_fft_mag(
            epochs=epochs,
            analysis_params=analysis_params
            )
    # Compute the snr over all fft magnitudes
    snr = snr_spectrum(
            fft_mag,
            analysis_params['noise_n_neighbor_bins'],
            analysis_params['noise_skip_neighbor_bins']
            )
    # Now slice SNR up to the desired frequency
    snr = snr[:, :, :len(freqs)]

    return snr, freqs


def compute_condition_psds_and_snrs(
        epochs,
        condition_name: str,
        parameters: dict,
        ROI: Optional[List[str]] = None
):
    """Compute PSDs and SNRs for a specific condition (left or right minimap).

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object for a specific condition.
    condition_name : str
        Name of the condition ('left_minimap' or 'right_minimap').
    parameters : dict
        Dictionary containing parameters for preprocessing and analysis.
    ROI : Optional[List[str]] 
        Optional list of channels to restrict the analysis to.
    Returns
    -------
    psds : dict
        Dictionary containing PSDs for each unique combination of parameters specific to the condition.
    snrs : dict
        Dictionary containing SNRs for each unique combination of parameters specific to the condition.
    freqs : ndarray
        Array of frequency values.
    """
    # Define each combination of frequencies and stimulus sides
    frequencies = ['3.75', '4.8']
    stimulus_sides = ['LEFT', 'RIGHT']
    # Initialize dictionaries to hold PSDs and SNRs
    psds = {}
    snrs = {}
    # Compute PSD and SNR for each frequency and stimulus side
    for frequency in frequencies:
        for side in stimulus_sides:
            # Construct the key to access specific PSDs and SNRs
            psd_key = f"{condition_name}_psds_{frequency.replace('.', '')}_{side.lower()}"
            snr_key = f"{condition_name}_snrs_{frequency.replace('.', '')}_{side.lower()}"
            epochs_key = f"{frequency}_{side}"
            # Compute PSD and SNR for the given epochs
            psds[psd_key], freqs = compute_psd(
                epochs[epochs_key], 
                parameters['preprocessing'],
                parameters['analysis'],
                ROI
            )
            snrs[snr_key] = snr_spectrum(
                psds[psd_key],
                parameters['analysis']['noise_n_neighbor_freqs'],
                parameters['analysis']['noise_skip_neighbor_freqs']
            )

    return psds, snrs, freqs
