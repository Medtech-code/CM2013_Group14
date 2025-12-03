import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.fft import fft
from scipy.signal import find_peaks, butter, filtfilt,spectrogram
import mne
import yasa
from sklearn.decomposition import PCA
from scipy import signal as sp_signal
from spectrum import arburg
from numpy.fft import rfftfreq
import pywt

def extract_time_domain_features(epoch,fs):
    """
    EXAMPLE: Extract basic time-domain features from a single epoch.

    This is a MINIMAL example with only 3 features.
    Students must implement the remaining 13+ time-domain features.

    Works for any signal type (EEG, EOG, EMG) but students should consider
    signal-specific features for optimal performance.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        dict: A dictionary of features.
    """
    # EXAMPLE: Only 3 basic features - students must add 13+ more
    features = {
        'mean': np.mean(epoch),
        'median': np.median(epoch),
        'std': np.std(epoch),
    }

    # TODO: Students must implement remaining time-domain features:
    # Basic statistical features:
    # 4
    features['rms'] = np.sqrt(np.mean(epoch**2))
    # 5
    features['min'] = np.min(epoch)
    # 6
    features['max'] = np.max(epoch)
    # 7
    features['range'] = np.max(epoch) - np.min(epoch)
    # 8
    features['skewness'] = scipy.stats.skew(epoch)
    # 9
    features['kurtosis'] = scipy.stats.kurtosis(epoch)

    # Signal complexity features:
    # 10
    features['zero_crossings'] = np.sum(np.diff(np.sign(epoch)) != 0)
    # 11
    features['hjorth_activity'] = np.var(epoch)
    # 12
    features['hjorth_mobility'] = np.sqrt(np.var(np.diff(epoch)) / np.var(epoch))
    # 13
    features['hjorth_complexity'] = hjorth_complexity(epoch)
    # 14
    features['q25'] = np.percentile(epoch, 25)
    # 15
    features['q75'] = np.percentile(epoch, 75)
    # 16
    features['iqr'] = features['q75'] - features['q25']



    return features

def extract_frequency_domain_features_welch(epoch, fs):


    f, S = compute_psd_welch(epoch, fs,nperseg=256)
    features={}
    
    # 15
    features['spectral_entropy'] = -np.sum((S/np.sum(S)) * np.log(S/np.sum(S) + 1e-12))
    # 16
    features['spectral_edge_freq_95'] = spectral_edge_frequency(epoch, fs, percent=0.95)
    
    # 17
    features['delta_power'] = band_power(f, S, (0.5, 4))
    # 18
    features['theta_power'] = band_power(f, S, (4, 8))
    # 19
    features['alpha_power'] = band_power(f, S, (8, 13))
    # 20
    features['beta_power'] = band_power(f, S, (13, 30))
    # 21
    features['gamma_power'] = band_power(f, S, (30, 50))

    # Absolute Band Power Ratios(as gibven in the paper "An Effective and Interpretable Sleep Stage Classification
    # Approach Using Multi-Domain Electroencephalogram and
    # Electrooculogram Features")

    # Relative Band Power Ratios
    # 22
    features['rbpr_delta_alpha'] = relative_band_power_ratio(f, S, (0.5, 4), (8, 13))

    # 23
    features['rbpr_theta_beta'] = relative_band_power_ratio(f, S, (4, 8), (13, 30))

    # 24
    # Slow fast ratio
    features['rbpr_delta_theta_alpha_beta'] =  (band_power(f, S, (0.5, 4))+band_power(f, S, (4, 8))) / (band_power(f, S, (8, 13))+band_power(f, S, (13, 30)))
    rel_powers = compute_relative_band_power(f, S,bands=None)
    features.update(rel_powers)
    return features    


def extract_frequency_domain_features_wavelet(epoch, wavelet='db4', level=5):
    # 23 features extracted
    coeffs = pywt.wavedec(epoch, wavelet=wavelet, level=level)
    A = coeffs[0]       # Approximation at level L
    D = coeffs[1:]      # Details D1..DL
    features = {}

    # Helper to compute energy and entropy
    def energy(c):
        return float(np.sum(c**2))
    def entropy(c):
        p = np.abs(c)
        s = np.sum(p) + 1e-12
        q = p / s
        return float(-np.sum(q * np.log(q + 1e-12)))

    for i, c in enumerate(D, start=1):
        # total features: 15-30
        features[f'D{i}_energy'] = energy(c)
        features[f'D{i}_entropy'] = entropy(c)
        features[f'D{i}_mean'] = float(np.mean(c))
        features[f'D{i}_std'] = float(np.std(c))
    # 31
    features['A5_energy'] = energy(A)
    # 32
    features['A5_entropy'] = entropy(A)

    # Aggregate EEG-band energies using mapping for fs=125
    beta_energy   = features['D2_energy']
    alpha_energy  = features['D3_energy']  # mixed alpha/sigma
    sigma_energy  = features['D3_energy']  # refine with CWT if needed
    theta_energy  = features['D4_energy']
    delta_energy  = features['D5_energy'] + features['A5_energy']

    # Ratios
    slow = delta_energy + theta_energy
    fast = alpha_energy + beta_energy
    features['slow_fast_ratio'] = float(slow / (fast + 1e-12))

    return features


def compute_psd_welch(signal, fs, nperseg=256):
    """
    Compute Power Spectral Density using Welch's method.
    Parameters:
    -----------
    signal : array-like
    Time-domain signal
    fs : float
    Sampling frequency in Hz (default: 256 Hz for EEG)
    nperseg : int
    Length of each segment for Welch's method
    Returns:
    --------
    freqs : array
    Frequency bins
    psd : array
    Power spectral density values
    """
    freqs, psd = sp_signal.welch(
    signal,
    fs=fs,
    window='hann',
    nperseg=nperseg,
    noverlap=nperseg//2,
    scaling='density'
    )
    return freqs, psd

def compute_psd_AR(x, fs, order, n_freqs=256):
    a, e, _ = arburg(x, order)
    freqs = rfftfreq(n_freqs*2, d=1/fs) 
    w = 2 * np.pi * freqs / fs
    k = np.arange(1, len(a))[:, None]  # shape (p, 1)
    Aw = 1.0 + (a[1:, None] * np.exp(-1j * w * k)).sum(axis=0)
    psd = e / (np.abs(Aw) ** 2)
    return freqs, psd

def ar_order_select_aic_bic(x, max_order=30):
    results = []
    N = len(x)
    for p in range(2, max_order + 1):
        a, e, _ = arburg(x, p)
        aic = N * np.log(e) + 2 * p
        bic = N * np.log(e) + p * np.log(N)
        results.append((p, e, aic, bic))

    p_aic = min(results, key=lambda r: r[2])[0]
    p_bic = min(results, key=lambda r: r[3])[0]
    return p_aic, p_bic, results


def integrate_band(freqs, psd, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return np.trapezoid(psd[mask], freqs[mask])

def band_power(f,Sxx, band):
    # f, Sxx = compute_psd_welch(epoch, fs)
    idx = np.logical_and(f >= band[0], f <= band[1])
    band_power = np.trapezoid(Sxx[idx], f[idx])
    return band_power

def compute_relative_band_power(f, Sxx, bands=None):
    """
    Compute relative (normalized) band powers for EEG bands.
    Args:
        f (np.ndarray): frequency array
        Sxx (np.ndarray): PSD values
        bands (dict): dictionary of bands, e.g. {'delta':(0.5,4), 'theta':(4,8), ...}
    Returns:
        dict: relative band powers
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta':  (13, 30),
            'gamma': (30, 50)
        }

    # Compute absolute band powers
    band_powers = {band: band_power(f, Sxx, rng) for band, rng in bands.items()}

    # Compute total power
    total_power = sum(band_powers.values()) + 1e-12  # avoid divide by zero

    # Normalize each band
    relative_powers = {f"{band}_relative": power / total_power
                       for band, power in band_powers.items()}

    return relative_powers


def spectral_edge_frequency(signal, fs, percent=0.95):
    """
    Compute Spectral Edge Frequency (SEF).
    Parameters:
    -----------
    percent : float
    Percentage of total power (e.g., 0.95 for SEF95)
    Returns:
    --------
    sef : float
    Frequency below which 'percent' of power is contained
    """
    freqs, psd = compute_psd_welch(signal, fs=fs)
    # Cumulative sum of power
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    # Find frequency where cumulative power reaches threshold
    threshold = percent * total_power
    idx = np.where(cumulative_power >= threshold)[0]
    if len(idx) > 0:
     sef = freqs[idx[0]]
    else:
     sef = freqs[-1]
    
    return sef


def relative_band_power_ratio(epoch, fs, band_num, band_den):
    power_num = band_power(epoch, fs, band_num)
    power_den = band_power(epoch, fs, band_den)
    return power_num / power_den


def extract_features(data, config,channel_info):
    """
    STUDENT IMPLEMENTATION AREA: Extract features based on current iteration.

    This function should handle both single-channel (old format) and
    multi-channel data (new format with 2 EEG + 2 EOG + 1 EMG channels).

    Iteration 1: 16 time-domain features per EEG channel
    Iteration 2: 31+ features (time + frequency domain) per channel
    Iteration 3: Multi-signal features (EEG + EOG + EMG)
    Iteration 4: Optimized feature set (selected subset)

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        config (module): The configuration module.

    Returns:
        np.ndarray: A 2D array of features (n_epochs, n_features).
    """
    print(f"Extracting features for iteration {config.CURRENT_ITERATION}...")
    fs=channel_info['eeg_fs']
    # Detect if we have multi-channel data structure
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG )")
        return extract_multi_channel_features(data, config,fs,debug=False)
    else:
        print("Processing single-channel data (backward compatibility)")
        return extract_single_channel_features(data,config,fs)


def extract_multi_channel_features(multi_channel_data, config,fs,debug=False):
    """
    Extract features from multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.

    Students should expand this significantly!
    """
    n_epochs = multi_channel_data['eeg'].shape[0]
    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = []

        # EEG features (2 channels)
        for ch in range(multi_channel_data['eeg'].shape[1]):
            eeg_signal = multi_channel_data['eeg'][epoch_idx, ch, :]
            eeg_time_domain_features = extract_time_domain_features(eeg_signal,fs)
            epoch_features.extend(list(eeg_time_domain_features.values()))
           
            # eeg_frequency_domain_features_welch = extract_frequency_domain_features_welch(eeg_signal, fs)
            # eeg_frequency_domain_features_wavelet = extract_frequency_domain_features_wavelet(eeg_signal, wavelet='db4', level=5)


            # eeg_frequency_domain_features = {}
            # eeg_frequency_domain_features.update(eeg_frequency_domain_features_welch)
            # eeg_frequency_domain_features.update(eeg_frequency_domain_features_wavelet)
            # epoch_features.extend(list(eeg_frequency_domain_features.values()))

        

        if config.CURRENT_ITERATION >= 3:
            # Add EOG features (2 channels)
            for ch in range(multi_channel_data['eog'].shape[1]):
                eog_signal = multi_channel_data['eog'][epoch_idx, ch, :]
                eog_features = extract_eog_features(eog_signal)
                epoch_features.extend(list(eog_features.values()))

                # Add EMG features (1 channel)
                emg_signal = multi_channel_data['emg'][epoch_idx, 0, :]
                emg_features = extract_emg_features(emg_signal)
                epoch_features.extend(list(emg_features.values()))

        all_features.append(epoch_features)

    features = np.array(all_features)
    scaler = RobustScaler()
    features = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    plt.scatter(proj[:,0], proj[:,1], cmap='viridis', alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2") 
    plt.title("PCA of Features")
    plt.colorbar(label="Class")
    plt.show()

    
    if config.CURRENT_ITERATION == 1:
        expected = 2 * 3  # 2 EEG channels × 3 features each
        print(f"Multi-channel Iteration 1: {features.shape[1]} features (target: {expected}+)")
        print("Students must implement remaining 13 time-domain features per EEG channel!")

    elif config.CURRENT_ITERATION >= 3:
        print(f"Multi-channel features extracted: {features.shape[1]} total")
        print("(2 EEG + 2 EOG + 1 EMG channels)")

    return features


def extract_single_channel_features(data, config,fs):
    """
    Backward compatibility for single-channel data.
    """
    if config.CURRENT_ITERATION == 1:
        # Iteration 1: Time-domain features (TARGET: 16 features)
        # CURRENT: Only 3 features implemented - students must add 13 more!
        all_features = []
        for epoch_index,epoch in enumerate(data):
            features = extract_time_domain_features(epoch,fs)
            # visualize_features(data,epoch_index)  # Visualize features for debugging
            all_features.append(list(features.values()))
        feature_names = list(extract_time_domain_features(data[0],fs).keys())   
        df_features = pd.DataFrame(all_features, columns=feature_names)
        scaler = RobustScaler()
        normalized_array = scaler.fit_transform(df_features)
        df_normalized = pd.DataFrame(normalized_array, columns=feature_names)
        all_features = df_normalized.values.tolist()
        features = np.array(all_features)     # 🔹 Show single-epoch visualization (first epoch by default)
        visualize_feature_distributions(features, feature_names)
        visualize_feature_trends(features, feature_names)


    elif config.CURRENT_ITERATION == 2:
        all_features = []
        feature_names = None


        num_epochs = len(data[0])  

        for i in range(num_epochs):
            eeg_epoch = data[0][i]
            eog_epoch = data[1][i]

            # Extract EEG features
            eeg_time = extract_time_domain_features(eeg_epoch, fs)
            eeg_welch = extract_frequency_domain_features_welch(eeg_epoch, fs)
            eeg_wavelet = extract_frequency_domain_features_wavelet(eeg_epoch, wavelet='db4', level=5)

            # Extract EOG features
            eog = extract_eog_features(eog_epoch, fs)

            # Combine all features
            epoch_features = {
                **eeg_time,
                **eeg_welch,
                **eeg_wavelet,
                **eog
            }

            # Set feature names ONCE from the FIRST epoch
            if feature_names is None:
                feature_names = list(epoch_features.keys())
                expected_len = len(feature_names)
            else:
                if len(epoch_features) != expected_len:
                    raise ValueError(
                        f"Feature count mismatch in epoch {i}: "
                        f"expected {expected_len}, got {len(epoch_features)}\n"
                        f"Missing: {set(feature_names) - set(epoch_features.keys())}\n"
                        f"Extra:   {set(epoch_features.keys()) - set(feature_names)}"
                    )

            all_features.append(list(epoch_features.values()))

        df_features = pd.DataFrame(all_features, columns=feature_names)

        scaler = RobustScaler()
        normalized_array = scaler.fit_transform(df_features)
        df_normalized = pd.DataFrame(normalized_array, columns=feature_names)
        df_normalized.to_csv("features.csv", index=False)
        all_features = df_normalized.values.tolist()
        features = np.array(all_features)   

    elif config.CURRENT_ITERATION >= 3:
        # TODO: Students must implement multi-signal features
        print("TODO: Students should use multi-channel data format for iteration 3+")
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))  # Empty features - students must implement

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return features


# def extract_eog_cross_channel_features(left_channel, right_channel, fs=100):
#     """
#     Extract cross-channel EOG features (horizontal saccades, correlation)
#     using left and right EOG channels.

#     Args:
#         left_channel (np.ndarray): Left EOG signal
#         right_channel (np.ndarray): Right EOG signal
#         fs (int): Sampling frequency

#     Returns:
#         dict: Features including cross-channel correlation and horizontal saccade count
#     """
#     left = np.array(left_channel)
#     right = np.array(right_channel)

#     # 1. Left-right correlation
#     # 30
#     lr_corr = np.corrcoef(left, right)[0, 1]

#     # # 2. Horizontal saccades (difference signal)
#     # diff_sig = left - right
#     # saccade_threshold = np.mean(np.abs(diff_sig)) + 2*np.std(diff_sig)
#     # saccade_peaks, _ = find_peaks(np.abs(diff_sig),
#     #                               height=saccade_threshold,
#     #                               distance=int(0.05*fs))
    
#     # if len(saccade_peaks) == 0:
#     #     horizontal_saccades = 0
#     # else:
#     #     horizontal_saccades = len(saccade_peaks)
    
#     features = {
#         'lr_correlation': lr_corr,
#         # 'horizontal_saccades':horizontal_saccades,
#         # 'saccade_peak_indices': saccade_peaks
#     }

#     return features


def extract_eog_features(eog_signal, fs=50):
    """
    Extract EOG-specific features for eye movement detection.

    If eog_signal is a dict with {'left':..., 'right':...}
    cross-channel correlation features will also be computed.
    """

    sig = np.array(eog_signal)
    
    # 35-37
    features = {
        # 1
        'eog_peak_amplitude': np.max(np.abs(eog_signal)),
        # 2
        'eog_var': np.var(sig),
        'eog_range': np.max(sig) - np.min(sig),
    }

    # 1. BLINK DETECTION 
    # Use adaptive threshold
    blink_threshold = np.mean(sig) + 2.5 * np.std(sig)

    blink_peaks, _ = find_peaks(np.abs(sig),
                                height=blink_threshold,
                                distance=int(0.1 * fs))  

    # 3
    features['blink_count'] = len(blink_peaks)
    # 4
    features['rem_score'] = len(blink_peaks) / len(eog_signal)


    # 2. RAPID EYE MOVEMENT (REM) FEATURES (0.5–5 Hz band)
    rem_band = bandpass(sig, 0.5, 5.0, fs)
    rem_energy = np.sum(rem_band ** 2)
    rem_zero_cross = np.sum(rem_band[:-1] * rem_band[1:] < 0)
    # 5
    features['rem_energy'] = rem_energy
    # 6
    features['rem_zero_crossings'] = rem_zero_cross

    # 3. SLOW EYE MOVEMENTS (SEM) FEATURES (< 0.5 Hz)
    sem_band = lowpass(sig, 0.5, fs)
    sem_variance = np.var(sem_band)
    sem_slope = np.mean(np.abs(np.diff(sem_band)))
    # 7
    features['sem_variance'] = sem_variance
    # 8
    features['sem_slope'] = sem_slope

    return features



def extract_emg_features(emg_signal,fs=125):
    """
    STUDENT TODO: Extract EMG-specific features for muscle tone detection.

    EMG signals are used to detect:
    - Muscle tone levels (high in wake, low in REM)
    - Muscle twitches and artifacts
    - Sleep-related muscle activity
    """
    features = {
        'emg_mean': np.mean(emg_signal),
        'emg_std': np.std(emg_signal),
        'emg_rms': np.sqrt(np.mean(emg_signal**2)),
    }

    # TODO: Students should add:
    # - High-frequency power (muscle activity indicator)
    # - Spectral edge frequency
    # - Muscle tone quantification

    # Signal power
    features['power'] = np.mean(emg_signal**2)
    
    # Variance
    features['variance'] = np.var(emg_signal)
    
    # High-frequency (20-40 Hz) power ratio
    f, Pxx = compute_psd_welch(emg_signal, fs=fs, nperseg=fs*2)  # PSD estimate
    band_power = np.trapezoid(Pxx[(f>=20) & (f<=40)], f[(f>=20) & (f<=40)])
    total_power = np.trapezoid(Pxx, f)
    features['hf_ratio'] = band_power / total_power if total_power > 0 else 0
    

    return features

def hjorth_complexity(epoch):
    """
    Calculate Hjorth Complexity of a signal epoch.

    Hjorth Complexity is a measure of the shape of the signal waveform,
    indicating how the frequency content changes over time.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        float: The Hjorth Complexity value.
    """
    first_deriv = np.diff(epoch)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(epoch)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    if var_zero == 0 or var_d1 == 0:
        return 0.0

    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility

    return complexity

def visualize_features(data, epoch_index, all_features=None, normalize=True):
    """
    Visualize a single epoch and its time-domain features, with optional dataset-wide normalization.

    Args:
        data (np.ndarray): 2D array, shape (n_epochs, n_samples)
        epoch_index (int): which epoch to visualize
        all_features (np.ndarray, optional): 2D array of shape (n_epochs, n_features)
            for computing global mean/std for normalization.
        normalize (bool): whether to normalize feature values for the bar chart
    """
    import numpy as np
    import matplotlib.pyplot as plt

    epoch = data[epoch_index]
    features = extract_time_domain_features(epoch)
    zero_crossings_idx = np.where(np.diff(np.sign(epoch)))[0]
    t = np.arange(len(epoch))

    feature_names = list(features.keys())
    feature_values = np.array(list(features.values()), dtype=float)

    # --- Normalization ---
    if normalize:
        if all_features is not None:
            # dataset-wide Z-score
            global_mean = np.mean(all_features, axis=0)
            global_std = np.std(all_features, axis=0)
            # prevent division by zero
            global_std[global_std == 0] = 1
            normalized_values = (feature_values - global_mean) / global_std
        else:
            # epoch-only Z-score
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            normalized_values = (feature_values - mean_val) / std_val if std_val != 0 else feature_values
    else:
        normalized_values = feature_values

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    #  Raw signal
    axes[0].plot(t, epoch, label='Epoch Signal')
    axes[0].axhline(features['mean'], color='r', linestyle='--', label=f"Mean = {features['mean']:.2e}")
    axes[0].scatter(zero_crossings_idx, epoch[zero_crossings_idx], color='orange', label='Zero Crossings', s=20)
    axes[0].set_title(f"Epoch {epoch_index} Signal with Key Features")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    #  Feature bar plot
    axes[1].bar(feature_names, normalized_values, color='skyblue')
    axes[1].set_title(
        f"Time-Domain Features of Epoch {epoch_index}" +
        (" (Normalized)" if normalize else "")
    )
    axes[1].set_ylabel("Feature Value (Z-score)" if normalize else "Feature Value")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def visualize_feature_distributions(features_array, feature_names):
    """
    Show feature distributions across all epochs.
    """
    df = pd.DataFrame(features_array, columns=feature_names)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, palette="Blues")
    plt.title("Time-Domain Feature Distribution Across Epochs")
    plt.xticks(rotation=45)
    plt.ylabel("Feature Value")
    plt.tight_layout()
    plt.show()
    


def visualize_feature_trends(features_array, feature_names):
    """
    Show how each feature evolves across epochs.
    """
    plt.figure(figsize=(14, 6))
    for i, name in enumerate(feature_names):
        plt.plot(features_array[:, i], label=name)
    plt.title("Time-Domain Feature Trends Across Epochs")
    plt.xlabel("Epoch Index")
    plt.ylabel("Feature Value")
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.show()


def review_outlier_epochs(data, features, feature_names, feature_key, fs=100, threshold=2.5):
    """
    Identify and visualize outlier epochs based on a selected feature.

    Args:
        data (np.ndarray): Raw signal data, shape (n_epochs, n_samples)
        features (np.ndarray): Feature matrix, shape (n_epochs, n_features)
        feature_names (list): List of feature names
        feature_key (str): Feature to use for outlier detection
        fs (int): Sampling frequency (Hz)
        threshold (float): Z-score threshold for outlier detection
    """
    # Convert to Z-scores
    feature_index = feature_names.index(feature_key)
    feature_values = features[:, feature_index]
    z_scores = (feature_values - np.mean(feature_values)) / np.std(feature_values)

    # Find outlier indices
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    print(f"Found {len(outlier_indices)} outlier epochs for feature '{feature_key}'")

    # Visualize each outlier
    for idx in outlier_indices:
        signal = data[idx]

        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        
        # Raw signal
        axs[0].plot(signal, color='blue')
        axs[0].set_title(f"Raw Signal - Epoch {idx} (Z-score: {z_scores[idx]:.2f})")
        axs[0].set_xlabel("Sample Index")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True)

        # Spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs)
        axs[1].pcolormesh(t, f, Sxx, shading='gouraud')
        axs[1].set_title("Spectrogram")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Frequency (Hz)")

        plt.tight_layout()
        plt.show()

def detect_blinks(data, threshold=None):
    if threshold is None:
        threshold = 2.5 * np.std(data)  # Adaptive threshold
    positive_peaks, _ = find_peaks(data, height=threshold)
    negative_peaks, _ = find_peaks(-data, height=threshold)
    blinks = np.sort(np.concatenate((positive_peaks, negative_peaks)))
    return blinks

def bandpass(signal, low, high, fs, order=2):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def lowpass(signal, cutoff, fs, order=2):
    b, a = butter(order, cutoff/(fs/2), btype='low')
    return filtfilt(b, a, signal)

def visualize_eog_peaks(
    eog_signal,
    fs,
    blink_peaks=None,
    saccade_peaks=None,
    rem_band=None,
    sem_band=None,
    title="EOG Signal with Detected Peaks"
):
    """
    Visualize EOG signal with detected events:
    - Blink peaks (from abs(sig))
    - Saccades (for left-right channel difference)
    - Optional: REM-band and SEM-band overlays

    Inputs:
        eog_signal    raw signal (1D numpy array)
        fs            sampling rate
        blink_peaks   indices of blink peaks
        saccade_peaks  indices of saccade peaks
        rem_band      filtered REM band (optional)
        sem_band      filtered SEM band (optional)
    """
    sig = np.array(eog_signal)
    t = np.arange(len(sig)) / fs

    plt.figure(figsize=(14, 5))
    plt.plot(t, sig, label="Raw EOG", alpha=0.8)

    # Blink Peaks
    if blink_peaks is not None:
        plt.scatter(
            t[blink_peaks],
            sig[blink_peaks],
            color="red",
            marker="x",
            s=70,
            label="Blink Peaks"
        )

    # Saccade Peaks
    if saccade_peaks is not None:
        plt.scatter(
            t[saccade_peaks],
            sig[saccade_peaks],
            color="green",
            marker="o",
            s=50,
            label="Saccade Peaks"
        )

    # Optional signals
    if rem_band is not None:
        plt.plot(t, rem_band, alpha=0.6, label="REM band (0.5–5 Hz)")

    if sem_band is not None:
        plt.plot(t, sem_band, alpha=0.6, label="SEM band (<0.5 Hz)")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def highpass_filter(signal, fs, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)



