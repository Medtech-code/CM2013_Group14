import numpy as np
import mne
import os
from pathlib import Path

try:
    from .xml_parser import parse_xml_annotations, create_epoch_labels
except ImportError:
    from xml_parser import parse_xml_annotations, create_epoch_labels


def load_training_data(edf_file_path, xml_file_path, epoch_length=30):

    print(f"Loading training data from {edf_file_path} and {xml_file_path}...")
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")
    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)

    recording_duration = raw.times[-1] 
    parsed_xml = parse_xml_annotations(xml_file_path)
    stages = parsed_xml['stages']
    n_epochs = int(recording_duration / epoch_length)
    labels = create_epoch_labels(stages, recording_duration, epoch_length)
    channel_names = raw.ch_names
    eog_channels = [ch for ch in channel_names if 'EOG' in ch.upper()]
    emg_channels = [ch for ch in channel_names if 'EMG' in ch.upper() or 'CHIN' in ch.upper()]
    eeg_candidates = []
    for ch in channel_names:
        ch_upper = ch.upper()
        if 'EEG' in ch_upper and 'SAO' not in ch_upper and 'SPO' not in ch_upper:
            eeg_candidates.append(ch)
    
        elif any(pattern in ch_upper for pattern in ['C3', 'C4', 'F3', 'F4', 'O1-', 'O2-', 'CZ', 'FZ', 'PZ']):
            eeg_candidates.append(ch)

    eeg_channels = [ch for ch in eeg_candidates
                    if ch not in eog_channels and ch not in emg_channels]

    print(f"Identified channels:")
    print(f"  EEG: {eeg_channels}")
    print(f"  EOG: {eog_channels}")
    print(f"  EMG: {emg_channels}")


    multi_channel_data = {}
    channel_info = {'epoch_length': epoch_length}

    if eeg_channels:
        eeg_raw = raw.copy().pick_channels(eeg_channels)
        eeg_data, eeg_fs = _extract_epochs(eeg_raw, epoch_length, n_epochs)
        multi_channel_data['eeg'] = eeg_data
        channel_info['eeg_names'] = eeg_channels
        channel_info['eeg_fs'] = eeg_fs
        print(f"  EEG: {eeg_data.shape[1]} channels, {eeg_data.shape[2]} samples/epoch, {eeg_fs} Hz")

    if eog_channels:
        eog_raw = raw.copy().pick_channels(eog_channels)
        eog_data, eog_fs = _extract_epochs(eog_raw, epoch_length, n_epochs)
        multi_channel_data['eog'] = eog_data
        channel_info['eog_names'] = eog_channels
        channel_info['eog_fs'] = eog_fs
        print(f"  EOG: {eog_data.shape[1]} channels, {eog_data.shape[2]} samples/epoch, {eog_fs} Hz")


    if emg_channels:
        emg_raw = raw.copy().pick_channels(emg_channels)
        emg_data, emg_fs = _extract_epochs(emg_raw, epoch_length, n_epochs)
        multi_channel_data['emg'] = emg_data
        channel_info['emg_names'] = emg_channels
        channel_info['emg_fs'] = emg_fs
        print(f"  EMG: {emg_data.shape[1]} channels, {emg_data.shape[2]} samples/epoch, {emg_fs} Hz")

    print(f"\nLoaded {n_epochs} epochs ({n_epochs*epoch_length/3600:.2f} hours)")
    _print_label_distribution(labels)

    labels = labels[:n_epochs]

    return multi_channel_data, labels, channel_info


def load_holdout_data(edf_file_path, epoch_length=30):

    print(f"Loading holdout data from {edf_file_path}...")
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")


    record_id = Path(edf_file_path).stem
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    recording_duration = raw.times[-1]
    n_epochs = int(recording_duration / epoch_length)
    channel_names = raw.ch_names

    eog_channels = [ch for ch in channel_names if 'EOG' in ch.upper()]
    emg_channels = [ch for ch in channel_names if 'EMG' in ch.upper() or 'CHIN' in ch.upper()]
    eeg_candidates = []

    channel_info = {'epoch_length': epoch_length}
    
    for ch in channel_names:
        ch_upper = ch.upper()
        if 'EEG' in ch_upper and 'SAO' not in ch_upper and 'SPO' not in ch_upper:
            eeg_candidates.append(ch)
        elif any(pattern in ch_upper for pattern in ['C3', 'C4', 'F3', 'F4', 'O1-', 'O2-', 'CZ', 'FZ', 'PZ']):
            eeg_candidates.append(ch)

    eeg_channels = [ch for ch in eeg_candidates
                    if ch not in eog_channels and ch not in emg_channels]

    print(f"Identified channels:")
    print(f"  EEG: {eeg_channels}")
    print(f"  EOG: {eog_channels}")
    print(f"  EMG: {emg_channels}")

    multi_channel_data = {}
    sampling_rates = {}

    if eeg_channels:
        eeg_raw = raw.copy().pick_channels(eeg_channels)
        eeg_data, eeg_fs = _extract_epochs(eeg_raw, epoch_length, n_epochs)
        multi_channel_data['eeg'] = eeg_data
        sampling_rates['eeg'] = eeg_fs
        channel_info['eeg_names'] = eeg_channels
        channel_info['eeg_fs'] = eeg_fs
        print(f"  EEG: {eeg_data.shape[1]} channels, {eeg_data.shape[2]} samples/epoch, {eeg_fs} Hz")

    if eog_channels:
        eog_raw = raw.copy().pick_channels(eog_channels)
        eog_data, eog_fs = _extract_epochs(eog_raw, epoch_length, n_epochs)
        multi_channel_data['eog'] = eog_data
        channel_info['eog_names'] = eog_channels
        channel_info['eog_fs'] = eog_fs
        sampling_rates['eog'] = eog_fs
        print(f"  EOG: {eog_data.shape[1]} channels, {eog_data.shape[2]} samples/epoch, {eog_fs} Hz")

    if emg_channels:
        emg_raw = raw.copy().pick_channels(emg_channels)
        emg_data, emg_fs = _extract_epochs(emg_raw, epoch_length, n_epochs)
        multi_channel_data['emg'] = emg_data
        channel_info['emg_names'] = emg_channels
        channel_info['emg_fs'] = emg_fs
        sampling_rates['emg'] = emg_fs
        print(f"  EMG: {emg_data.shape[1]} channels, {emg_data.shape[2]} samples/epoch, {emg_fs} Hz")

    record_info = {
        'record_id': record_id,
        'n_epochs': n_epochs,
        'channels': eeg_channels + eog_channels + emg_channels,
        'sampling_rates': sampling_rates,
        'epoch_length': epoch_length
    }

    print(f"Loaded {n_epochs} epochs ({n_epochs*epoch_length/3600:.2f} hours)")

    return multi_channel_data, record_info, channel_info


def _extract_epochs(raw, epoch_length, n_epochs):

    data = raw.get_data()  
    fs = raw.info['sfreq']
    n_channels = data.shape[0]
    samples_per_epoch = int(epoch_length * fs)
    total_samples_needed = n_epochs * samples_per_epoch
    if data.shape[1] > total_samples_needed:
        data = data[:, :total_samples_needed]
    elif data.shape[1] < total_samples_needed:
        padding = total_samples_needed - data.shape[1]
        data = np.pad(data, ((0, 0), (0, padding)), mode='constant')

    epochs = data.reshape(n_channels, n_epochs, samples_per_epoch)
    epochs = np.transpose(epochs, (1, 0, 2)) 
    return epochs, fs


def _print_label_distribution(labels):
  
    unique, counts = np.unique(labels, return_counts=True)
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    print("Sleep stage distribution:")
    for stage, count in zip(unique, counts):
        if stage < len(stage_names):
            pct = (count / len(labels)) * 100
            print(f"  {stage_names[stage]}: {count} epochs ({pct:.1f}%)")


