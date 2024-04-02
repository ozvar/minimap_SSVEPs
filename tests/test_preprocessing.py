from src.preprocessing import * 
import pytest


def test_get_all_sample_filenames_happy_path(mocker):

    expected_files = ['data1.edf', 'data2.edf']
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('glob.glob', return_value=expected_files)
    assert get_all_sample_filenames('fake_directory', 'edf') == expected_files


def test_get_all_sample_filenames_no_directory(mocker):
    
    mocker.patch('glob.glob', side_effect=Exception('Directory not found'))
    with pytest.raises(Exception) as exc_info:
        get_all_sample_filenames('fake_directory', 'edf')
    assert str(exc_info.value) == 'Directory not found'


def test_get_all_sample_filenames_no_matches(mocker):

    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('glob.glob', return_value=[])
    with pytest.raises(Exception) as exc_info:
        get_all_sample_filenames('fake_directory', 'edf')
    assert str(exc_info.value) == 'No files with matching extension found in directory'


def test_get_all_sample_filenames_permission_denied(mocker):

    mocker.patch('src.utils.os.path.isdir', side_effect=PermissionError("Permission denied"))
    with pytest.raises(PermissionError) as exc_info:
        get_all_sample_filenames('restricted_directory', 'edf')
    assert "Permission denied" in str(exc_info.value)


def test_read_data(mocker):

    mock_read_raw_edf = mocker.patch('mne.io.read_raw_edf', return_value='mocked_data')
    result = read_data('fake_file', preload=True)

    mock_read_raw_edf.assert_called_once_with('fake_file', preload=True)
    assert result == 'mocked_data'


def test_set_montage(mocker):

    mock_data = mocker.MagicMock()
    set_montage(mock_data)
    mock_data.set_montage.assert_called_once_with(
        montage='standard_1005', match_case=False, on_missing='warn', match_alias=True
    )


def test_filter_data_applies_filter_correctly(mocker):

    mock_data = mocker.MagicMock()
    filter_data(mock_data, l_freq=0.5, h_freq=100)
    mock_data.filter.assert_called_once_with(l_freq=0.5, h_freq=100)


def test_filter_data_drops_channels_correctly(mocker):

    mock_data = mocker.MagicMock()
    channels_to_drop = ['EMG', 'EKG', 'non-existent-channel']
    filter_data(mock_data, l_freq=0.5, h_freq=100, channels_to_drop=channels_to_drop)

    expected_calls = [mocker.call([channel]) for channel in channels_to_drop]
    # Adjust expected calls to reflect the chaining of the filter and drop_channels methods
    expected_calls = [mocker.call.filter(l_freq=0.5, h_freq=100)] + \
                     [mocker.call.filter().drop_channels([channel]) for channel in channels_to_drop]
    assert mock_data.mock_calls == expected_calls


def test_filter_data_handles_nonexistent_channels_gracefully(mocker):

    mock_data = mocker.MagicMock()
    # Simulate ValueError for non-existent channels
    channels_to_drop = ['NON_EXISTENT_CHANNEL']
    mock_data.drop_channels.side_effect = ValueError(f"Channel {channels_to_drop} not found")
    try:
        filter_data(mock_data, l_freq=0.5, h_freq=100, channels_to_drop=channels_to_drop)
    except ValueError:
        pytest.fail("filter_data should handle ValueError for non-existent channels gracefully.")


def test_get_epochs_sets_montage_correctly(mocker):

    mock_data = mocker.MagicMock()
    mocker.patch('mne.find_events', return_value=[])
    mocker.patch('mne.Epochs', return_value=mocker.MagicMock())

    get_epochs(mock_data)
    mock_data.set_montage.assert_called_once_with(
        montage='standard_1005', match_case=False, on_missing='warn', match_alias=True
    )


def test_get_epochs_find_events_called_correctly(mocker):

    mock_data = mocker.MagicMock()
    mock_find_events = mocker.patch('mne.find_events', return_value=[])
    mocker.patch('mne.Epochs', return_value=mocker.MagicMock())

    get_epochs(mock_data)
    mock_find_events.assert_called_once_with(mock_data)


def test_get_epochs_create_epochs_called_correctly(mocker):

    mock_data = mocker.MagicMock()
    mock_events = mocker.patch('mne.find_events', return_value=[[1, 0, 2]])
    mock_epochs_constructor = mocker.patch('mne.Epochs', return_value=mocker.MagicMock())

    get_epochs(mock_data)    
    mock_epochs_constructor.assert_called_once_with(
        mock_data,
        mock_events.return_value,
        event_id={'4.8_LEFT': 65286, '3.75_LEFT': 65284, '4.8_RIGHT': 65289, '3.75_RIGHT': 65287},
        tmin=1.1,
        tmax=11.1,
        baseline=None,
        preload=True
    )


def test_get_epochs_resample_called_correctly(mocker):

    mock_data = mocker.MagicMock()
    mocker.patch('mne.find_events', return_value=[])
    mock_epochs = mocker.MagicMock()
    mocker.patch('mne.Epochs', return_value=mock_epochs)

    get_epochs(mock_data)
    mock_epochs.resample.assert_called_once_with(sfreq=100)
