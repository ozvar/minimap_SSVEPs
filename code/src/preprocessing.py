import glob
import mne
import os
import random
import numpy as np
from typing import List, Tuple


def get_all_sample_filenames(
    samples_dir: str,
    extension: str = 'edf'
):
    # Ensure directory exists
    if not os.path.isdir(samples_dir):
        raise Exception('Directory not found')
    all_filenames = [i for i in glob.glob(os.path.join(samples_dir, f'*.{extension}'))]
    # Ensure there are files in the directory
    if not all_filenames:
        raise Exception('No files with matching extension found in directory')
    return all_filenames


def read_data(
    file_path: str,
    preload : bool = True
):
    """
    This function exists primarily because we want to keep the preload default to True.
    We can adjust this later if we want to add more parameters or operations to the reading process
    """
    data_out = mne.io.read_raw_edf(file_path, preload = preload)
    return data_out


def filter_data(
    data,
    l_freq: float = 0.5,
    h_freq: int = 100,
    channels_to_drop: List[str] = ['EMG', 'EKG', 'VEO', 'HEO',  'HEOL', 'HEOR', 'M1', 'M2']
):
    """
    This function drops channels that have either disproportionate values or that don't exist.
    """
    data = data.filter(l_freq=l_freq, h_freq=h_freq)

    for channel in channels_to_drop:
        try:
            # MNE expects an iterable of channel names
            data.drop_channels([channel])
        except ValueError:
            print(f"Channel {channel} not found")
            pass
    return data


def set_reference(
    data,
    reference: str = 'average'
):
    data.set_eeg_reference(reference)
    return data
        

def get_epochs(
    data,
    tmin: float,
    tmax: float,
    montage: str = 'standard_1005',
    match_case: bool = False,
    on_missing: str = 'warn',
    match_alias: bool = True,
    baseline: Tuple[float, float] = None,
    resample_freq: int = 100,
    event_id: dict = {'4.8_LEFT': 65286, '3.75_LEFT': 65284, '4.8_RIGHT': 65289, '3.75_RIGHT': 65287}
):
    # Just realised, are these inplace functions? not ideal
    data.set_montage(
        montage = montage,
        match_case = match_case,
        on_missing = on_missing,
        match_alias = match_alias
    )
    events = mne.find_events(data)
    epochs = mne.Epochs(
            data,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True)
    epochs = epochs.resample(sfreq = resample_freq)
    return epochs


def ingest_sample(
    sample_filename: str,
    preprocessing_params: dict
):
    print(sample_filename)
    data = read_data(sample_filename)
    filter_params = {k: v for k, v in preprocessing_params.items() if k in ['l_freq', 'h_freq', 'channels_to_drop']}
    data = filter_data(data, **filter_params)
    epoch_params = {k: v for k, v in preprocessing_params.items() if k in ['resample_freq', 'baseline', 'tmin', 'tmax']}
    if preprocessing_params['reference_channels'] is not None:
        data = set_reference(data, reference = preprocessing_params['reference_channels'] )
    epochs = get_epochs(data, **epoch_params)
    return epochs


def ingest_samples(
    sample_filenames: List[str],
    preprocessing_params: dict
):
    epochs_out = [None] * len(sample_filenames)
    for count, filename in enumerate(sample_filenames):
        epochs_out[count] = ingest_sample(filename, preprocessing_params)
    epochs_out = mne.concatenate_epochs(epochs_out)
    return epochs_out


def equalize_epoch_counts(
    left_epochs,
    right_epochs
):
    """
    Equalizes the number of epochs for each event between two sets of epochs
    by randomly removing excess epochs from one of the sets.
    
    Parameters:
    - left_epochs: Epochs object for the left condition.
    - right_epochs: Epochs object for the right condition.
    """
    common_events = set(left_epochs.events[:, 2]) & set(right_epochs.events[:, 2])

    for event in common_events:
        left_count = np.sum(left_epochs.events[:, 2] == event)
        right_count = np.sum(right_epochs.events[:, 2] == event)
        # Determine the number of epochs to remove to equalize counts
        excess = abs(left_count - right_count)
        
        if left_count > right_count:
            to_remove = random.sample(list(np.where(left_epochs.events[:, 2] == event)[0]), excess)
            left_epochs.drop(to_remove)
        elif right_count > left_count:
            to_remove = random.sample(list(np.where(right_epochs.events[:, 2] == event)[0]), excess)
            right_epochs.drop(to_remove)
