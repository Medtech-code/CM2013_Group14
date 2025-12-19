from scipy.signal import butter, lfilter, iirnotch, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import welch
import numpy as np

def bandpass_filter(data, low_cutoff, high_cutoff, fs):

    nyquist = 0.5 * fs
    b, a = butter(4, [low_cutoff/nyquist, high_cutoff/nyquist], btype='band', analog=False)
    padlen = 3 * 4
    y = filtfilt(b, a, data, padtype='even', padlen=padlen)
    return y

def notch_filter(data, notch_freq, eeg_fs):
    quality_factor = 30  
    b, a = iirnotch(notch_freq, quality_factor, eeg_fs)
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal


def preprocess(data, config, channel_info):

    print(f"Preprocessing data for iteration {config.CURRENT_ITERATION}...")
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:

        return preprocess_multi_channel(data, config, channel_info)
    else:
        print("Processing single-channel data (backward compatibility)")
        return preprocess_single_channel(data, config)


def preprocess_multi_channel(multi_channel_data, config, channel_info):

    preprocessed_data = {}
    print("preprocess_multo_channel function")
    eeg_data = multi_channel_data['eeg']
    eeg_fs = channel_info['eeg_fs'] 
    preprocessed_eeg = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[1]):
        print(f"Processing channel 1...")
        for epoch in range(eeg_data.shape[0]):
            signal = eeg_data[epoch, ch, :]
            filtered_signal = bandpass_filter(signal, config.HIGH_PASS_FILTER_FREQ, config.LOW_PASS_FILTER_FREQ, eeg_fs)
            for notch_freq in [50, 100, 150]:
                if notch_freq < eeg_fs / 2:
                    filtered_signal = notch_filter(filtered_signal, notch_freq, eeg_fs)
 
            if ch == 0 and epoch == 0:
                print("Running validation for first EEG epoch...")
                validate_filtering(signal, filtered_signal, eeg_fs) 

            preprocessed_eeg[epoch, ch, :] = filtered_signal


    if config.CURRENT_ITERATION >= 2: 
        eog_data = multi_channel_data['eog']
        eog_fs = channel_info['eog_fs']   
        preprocessed_eog = np.zeros_like(eog_data)

        for ch in range(eog_data.shape[1]):
            for epoch in range(eog_data.shape[0]):
                signal = eog_data[epoch, ch, :]
                filtered_signal_eog = bandpass_filter(signal, 0.5, 40, eog_fs) 
                preprocessed_eog[epoch, ch, :] = filtered_signal_eog
        
        filtered_snapshot = preprocessed_eeg.copy() 
    
        for epoch in range(eeg_data.shape[0]):
                eeg_epoch = preprocessed_eeg[epoch, :, :].T  
                eog_epoch = preprocessed_eog[epoch, :, :].T  
                try:
                    b = np.linalg.solve(eog_epoch.T @ eog_epoch, eog_epoch.T @ eeg_epoch)
                except np.linalg.LinAlgError:
                    b = np.linalg.pinv(eog_epoch.T @ eog_epoch) @ eog_epoch.T @ eeg_epoch
                eeg_corrected = eeg_epoch - eog_epoch @ b
                preprocessed_eeg[epoch, :, :] = eeg_corrected.T

        preprocessed_data['eeg'] = preprocessed_eeg
        preprocessed_data['eog'] = preprocessed_eog
        fig, axes = plt.subplots(2, 1, figsize=(15, 4))
        axes[0].plot(filtered_snapshot[epoch, 0, :500], color='black', label='Filtered eeg Channel 0 - (after bandpass+notch)')
        axes[0].plot(eeg_epoch[:500, 0], color='gray', label='Artifact-corrected EEG Channel 0')
        axes[0].legend()
        axes[1].plot(filtered_snapshot[epoch, 0, :500], color='black', label='Raw Channel 1 - (after bandpass+notch)')
        axes[1].plot(eeg_epoch[:500, 1], color='gray', label='Artifact-corrected EEG Channel 1')
        axes[1].legend()
        plt.xlabel('Sample')
        plt.tight_layout()
        plt.show()


    if config.CURRENT_ITERATION >= 3: 
     
        emg_data = multi_channel_data['emg']
        emg_fs = channel_info['emg_fs']  
        preprocessed_emg = np.zeros_like(emg_data)
        all_powers = []

        for epoch in range(emg_data.shape[0]):
            signal = emg_data[epoch, 0, :]
            filtered_emg = bandpass_filter(signal, 20, 60, emg_fs)
            preprocessed_emg[epoch, 0, :] = filtered_emg
            f_emg, Pxx_emg = welch(filtered_emg, fs=emg_fs)
            band_mask = (f_emg >= 20) & (f_emg <= 40)
            power_20_40 = np.trapezoid(Pxx_emg[band_mask], f_emg[band_mask])
            all_powers.append(power_20_40)
            if epoch < 5:
                print("epoch", epoch, "power_20_40 =", power_20_40)
        threshold = np.percentile(all_powers, 60)
        print("EMG power threshold:", threshold)

        for epoch in range(emg_data.shape[0]):
            signal = emg_data[epoch, 0, :]
            filtered_emg = bandpass_filter(signal, 20, 60, emg_fs)
            preprocessed_emg[epoch, 0, :] = filtered_emg

            f_emg, Pxx_emg = welch(filtered_emg, fs=emg_fs)
            band_mask = (f_emg >= 20) & (f_emg <= 40)
            power_20_40 = np.trapezoid(Pxx_emg[band_mask], f_emg[band_mask])

            if power_20_40 > threshold:
    
                for ch in range(preprocessed_eeg.shape[1]):
                    eeg_epoch_ch = preprocessed_eeg[epoch, ch, :]
                    eeg_strong_lp = bandpass_filter(eeg_epoch_ch,
                                                    config.HIGH_PASS_FILTER_FREQ,
                                                    20,
                                                    eeg_fs)
                    preprocessed_eeg[epoch, ch, :] = eeg_strong_lp

        preprocessed_data['emg'] = preprocessed_emg
        print("Multi-channel preprocessing applied to EEG + EOG + EMG")
    elif config.CURRENT_ITERATION >= 2:
        print("Iteration 2: Processing EEG + EOG channels")
    else:
        print("Iteration 1: Processing EEG channels only")


    return preprocessed_data


def preprocess_single_channel(data, config):
  
    if config.CURRENT_ITERATION == 1:
        fs = 125 
        preprocessed_data = np.zeros_like(data) 
        for epoch in range(data.shape[0]):
            signal = data[epoch,:]
            filtered = bandpass_filter(signal, config.HIGH_PASS_FILTER_FREQ, config.LOW_PASS_FILTER_FREQ, fs)
            for notch_freq in [50, 100, 150]:
                if notch_freq < fs / 2:
                    filtered = notch_filter(filtered, notch_freq, fs)
            preprocessed_data[epoch,:] = filtered
            if epoch == 0:
                print("Running validation for first EEG epoch...")
                validate_filtering(signal, filtered, fs)
                
    elif config.CURRENT_ITERATION == 2:
        preprocessed_data = data 

    elif config.CURRENT_ITERATION >= 3:
        preprocessed_data = data 

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return preprocessed_data


def validate_filtering(original, filtered, fs):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    if original.ndim > 1:
        original = original[0]
    if filtered.ndim > 1:
        filtered = filtered[0]

    print("Original stats: mean={:.4e}, min={:.4e}, max={:.4e}".format(np.mean(original), np.min(original), np.max(original)))
    print("Filtered stats: mean={:.4e}, min={:.4e}, max={:.4e}".format(np.mean(filtered), np.min(filtered), np.max(filtered)))
    mean_before = np.mean(original)
    mean_after = np.mean(filtered)
    print(f"Mean before: {mean_before:.4f}, Mean after: {mean_after:.4f}")

    try:
        corr = np.corrcoef(original, filtered)[0, 1]
    except Exception:
        corr = float('nan')
    print(f"Correlation between original and filtered: {corr:.4f}")

    plt.figure(figsize=(12,5))
    plt.plot(original, label='Original')
    plt.plot(filtered, label='Filtered')
    plt.title('Time-domain Signal: Original vs Filtered')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    f_orig, Pxx_orig = welch(original, fs)
    f_filt, Pxx_filt = welch(filtered, fs)
    plt.figure(figsize=(10, 5))
    plt.semilogy(f_orig, Pxx_orig, label="Original")
    plt.semilogy(f_filt, Pxx_filt, label="Filtered")
    plt.title("PSD Comparison (Full Band)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.semilogy(f_orig, Pxx_orig, label='Original')
    plt.semilogy(f_filt, Pxx_filt, label='Filtered')
    plt.xlim(0, 5)
    plt.title('PSD: Delta Band Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.show()


    plt.figure(figsize=(8, 4))
    plt.semilogy(f_orig, Pxx_orig, label='Original')
    plt.semilogy(f_filt, Pxx_filt, label='Filtered')
    plt.xlim(45, 55)
    plt.title('PSD: 50 Hz Region Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.show()


    notch_band = (f_orig > 45) & (f_orig < 55)
    power_orig = np.sum(Pxx_orig[notch_band])
    power_filt = np.sum(Pxx_filt[notch_band])
    eps = 1e-12
    if power_orig < eps:
        print("WARNING: No original power at 50 Hz band; skipping ratio calculation.")
        power_ratio_50Hz = 0
    else:
        power_ratio_50Hz = power_filt / power_orig
    print(f"Powerline noise (50Hz band) reduced to {power_ratio_50Hz * 100:.2f}% of original")

    if abs(mean_after) < 0.05 and (abs(corr) > 0.8) and (power_ratio_50Hz < 0.2):
        print("Validation PASSED")
    else:
        print("Validation WARN  Some artifacts remain.")

lowpass_filter = bandpass_filter