import config
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
import os
output_dir = f"result/iteration_{config.CURRENT_ITERATION}"
os.makedirs(output_dir, exist_ok=True)
def extract_time_domain_features(epoch,fs):

    features = {
        'mean': np.mean(epoch),
        'median': np.median(epoch),
        'std': np.std(epoch),
    }
    features['rms'] = np.sqrt(np.mean(epoch**2))
    features['min'] = np.min(epoch)
    features['max'] = np.max(epoch)
    features['range'] = np.max(epoch) - np.min(epoch)
    features['skewness'] = scipy.stats.skew(epoch)
    features['kurtosis'] = scipy.stats.kurtosis(epoch)
    features['zero_crossings'] = np.sum(np.diff(np.sign(epoch)) != 0)
    features['hjorth_activity'] = np.var(epoch)
    features['hjorth_mobility'] = np.sqrt(np.var(np.diff(epoch)) / np.var(epoch))
    features['hjorth_complexity'] = hjorth_complexity(epoch)
    features['q25'] = np.percentile(epoch, 25)
    features['q75'] = np.percentile(epoch, 75)
    features['iqr'] = features['q75'] - features['q25']

    return features

def extract_frequency_domain_features_welch(epoch, fs):


    f, S = compute_psd_welch(epoch, fs,nperseg=256)
    features={}
    
    features['spectral_entropy'] = -np.sum((S/np.sum(S)) * np.log(S/np.sum(S) + 1e-12))
    features['spectral_edge_freq_95'] = spectral_edge_frequency(epoch, fs, percent=0.95)
    features['delta_power'] = band_power(f, S, (0.5, 4))
    features['theta_power'] = band_power(f, S, (4, 8))
    features['alpha_power'] = band_power(f, S, (8, 13))
    features['beta_power'] = band_power(f, S, (13, 30))
    features['gamma_power'] = band_power(f, S, (30, 50))
    features['rbpr_delta_alpha'] = relative_band_power_ratio(f, S, (0.5, 4), (8, 13))
    features['rbpr_theta_beta'] = relative_band_power_ratio(f, S, (4, 8), (13, 30))
    features['rbpr_delta_theta_alpha_beta'] =  (band_power(f, S, (0.5, 4))+band_power(f, S, (4, 8))) / (band_power(f, S, (8, 13))+band_power(f, S, (13, 30)))
    rel_powers = compute_relative_band_power(f, S,bands=None)
    features.update(rel_powers)

    return features    


def extract_frequency_domain_features_wavelet(epoch, wavelet='db4', level=5):

    coeffs = pywt.wavedec(epoch, wavelet=wavelet, level=level)
    A = coeffs[0]       
    D = coeffs[1:]    
    features = {}

    def energy(c):
        return float(np.sum(c**2))
    def entropy(c):
        p = np.abs(c)
        s = np.sum(p) + 1e-12
        q = p / s
        return float(-np.sum(q * np.log(q + 1e-12)))

    for i, c in enumerate(D, start=1):
        features[f'D{i}_energy'] = energy(c)
        features[f'D{i}_entropy'] = entropy(c)
        features[f'D{i}_mean'] = float(np.mean(c))
        features[f'D{i}_std'] = float(np.std(c))

    features['A5_energy'] = energy(A)
    features['A5_entropy'] = entropy(A)
    beta_energy   = features['D2_energy']
    alpha_energy  = features['D3_energy']  
    sigma_energy  = features['D3_energy'] 
    theta_energy  = features['D4_energy']
    delta_energy  = features['D5_energy'] + features['A5_energy']
    slow = delta_energy + theta_energy
    fast = alpha_energy + beta_energy
    features['slow_fast_ratio'] = float(slow / (fast + 1e-12))

    return features


def compute_psd_welch(signal, fs, nperseg=256):

    freqs, psd = sp_signal.welch(
    signal,
    fs=fs,
    window='hann',
    nperseg=nperseg,
    noverlap=nperseg//2,
    scaling='density'
    )
    return freqs, psd


def band_power(f,Sxx, band):
    idx = np.logical_and(f >= band[0], f <= band[1])
    band_power = np.trapezoid(Sxx[idx], f[idx])
    return band_power

def compute_relative_band_power(f, Sxx, bands=None):

    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta':  (13, 30),
            'gamma': (30, 50)
        }

    band_powers = {band: band_power(f, Sxx, rng) for band, rng in bands.items()}
    total_power = sum(band_powers.values()) + 1e-12 
    relative_powers = {f"{band}_relative": power / total_power
                       for band, power in band_powers.items()}

    return relative_powers


def spectral_edge_frequency(signal, fs, percent=0.95):

    freqs, psd = compute_psd_welch(signal, fs=fs)
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
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
   
    print(f"Extracting features for iteration {config.CURRENT_ITERATION}...")
    fs_eeg=channel_info['eeg_fs']
    fs_eog=channel_info['eog_fs']
    fs_emg=channel_info['emg_fs']
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG (+ EMG))")
        return extract_multi_channel_features(data, config,fs_eeg,fs_eog,fs_emg,debug=False)
    else:
        print("Processing single-channel data (backward compatibility)")
        return extract_single_channel_features(data,config,fs_eeg,fs_eog,fs_emg)


def extract_multi_channel_features(multi_channel_data, config,fs_eeg,fs_eog,fs_emg, debug=False):

    n_epochs = multi_channel_data['eeg'].shape[0]
    all_features = []

    time_names   = list(extract_time_domain_features(multi_channel_data['eeg'][0, 0, :], fs_eeg).keys())
    welch_names  = list(extract_frequency_domain_features_welch(multi_channel_data['eeg'][0, 0, :], fs_eeg).keys())
    wavelet_names = list(extract_frequency_domain_features_wavelet(multi_channel_data['eeg'][0, 0, :], wavelet='db4', level=5).keys())
    eog_names    = list(extract_eog_features(multi_channel_data['eog'][0, 0, :],fs_eog).keys())
    emg_names    = list(extract_emg_features(multi_channel_data['emg'][0, 0, :],fs_eog).keys())

    all_feature_names = []
    for ch in range(multi_channel_data['eeg'].shape[1]):
        all_feature_names += [f"EEG_ch{ch}_{k}" for k in time_names]
        all_feature_names += [f"EEG_ch{ch}_{k}" for k in welch_names]
        all_feature_names += [f"EEG_ch{ch}_{k}" for k in wavelet_names]

    if config.CURRENT_ITERATION >= 3:
        for ch in range(multi_channel_data['eog'].shape[1]):
            all_feature_names += [f"EOG_ch{ch}_{k}" for k in eog_names]

        all_feature_names += [f"EMG_{k}" for k in emg_names]

    for epoch_idx in range(n_epochs):
        epoch_features = []

        for ch in range(multi_channel_data['eeg'].shape[1]):
            eeg_signal = multi_channel_data['eeg'][epoch_idx, ch, :]

            epoch_features += list(extract_time_domain_features(eeg_signal, fs_eeg).values())
            epoch_features += list(extract_frequency_domain_features_welch(eeg_signal, fs_eeg).values())
            epoch_features += list(extract_frequency_domain_features_wavelet(eeg_signal, wavelet='db4', level=5).values())

        if config.CURRENT_ITERATION >= 3:
            for ch in range(multi_channel_data['eog'].shape[1]):
                eog_signal = multi_channel_data['eog'][epoch_idx, ch, :]
                epoch_features += list(extract_eog_features(eog_signal,fs_eog).values())

            emg_signal = multi_channel_data['emg'][epoch_idx, 0, :]
            epoch_features += list(extract_emg_features(emg_signal,fs_emg).values())

        all_features.append(epoch_features)


    features = np.array(all_features)
    df_features = pd.DataFrame(features, columns=all_feature_names)
    csv_filename = os.path.join(
    output_dir,
    f"features_iter{config.CURRENT_ITERATION}.csv")
    df_features.to_csv(csv_filename, index=False)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], cmap='viridis', alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Features")
    plt.show()

    
    if config.CURRENT_ITERATION == 1:
        expected = 2 * 3  
        print(f"Multi-channel Iteration 1: {features.shape[1]} features (target: {expected}+)")
        print("Students must implement remaining 13 time-domain features per EEG channel!")

    elif config.CURRENT_ITERATION >= 3:
        print(f"Multi-channel features extracted: {features.shape[1]} total")
        print("(2 EEG + 2 EOG + 1 EMG channels)")

    return features, all_feature_names


def extract_single_channel_features(data, config,fs_eeg,fs_eog,fs_emg):

    if config.CURRENT_ITERATION == 1:
        all_features = []
        for epoch_index,epoch in enumerate(data):
            features = extract_time_domain_features(epoch,fs_eeg)
            all_features.append(list(features.values()))
        feature_names = list(extract_time_domain_features(data[0],fs_eeg).keys())   
        df_features = pd.DataFrame(all_features, columns=feature_names)
        csv_filename = os.path.join(
        output_dir,
        f"features_iter{config.CURRENT_ITERATION}.csv")
        df_features.to_csv(csv_filename, index=False)
        all_features = df_features.values.tolist()
        features = np.array(all_features)   
        visualize_feature_distributions(features, feature_names)
        visualize_feature_trends(features, feature_names)


    elif config.CURRENT_ITERATION == 2:
        all_features = []
        feature_names = None


        num_epochs = len(data[0])  

        for i in range(num_epochs):
            eeg_epoch = data[0][i]
            eog_epoch = data[1][i]

            eeg_time = extract_time_domain_features(eeg_epoch, fs_eeg)
            eeg_welch = extract_frequency_domain_features_welch(eeg_epoch, fs_eeg)
            eeg_wavelet = extract_frequency_domain_features_wavelet(eeg_epoch, wavelet='db4', level=5)
            eog = extract_eog_features(eog_epoch, fs_eog)
            epoch_features = {
                **eeg_time,
                **eeg_welch,
                **eeg_wavelet,
                **eog
            }

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
        csv_filename = os.path.join(
        output_dir,
        f"features_iter{config.CURRENT_ITERATION}.csv")
        df_features.to_csv(csv_filename, index=False)
        all_features = df_features.values.tolist()
        features = np.array(all_features)   

    elif config.CURRENT_ITERATION >= 3:
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))  

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return features, feature_names


def extract_eog_features(eog_signal, fs):
    sig = np.array(eog_signal)
    features = {

        'eog_peak_amplitude': np.max(np.abs(eog_signal)),
        'eog_var': np.var(sig),
        'eog_range': np.max(sig) - np.min(sig),
    }
    blink_threshold = np.mean(sig) + 2.5 * np.std(sig)

    blink_peaks, _ = find_peaks(np.abs(sig),
                                height=blink_threshold,
                                distance=int(0.1 * fs))  

    features['blink_count'] = len(blink_peaks)
    features['rem_score'] = len(blink_peaks) / len(eog_signal)
    rem_band = bandpass(sig, 0.5, 5.0, fs)
    rem_energy = np.sum(rem_band ** 2)
    rem_zero_cross = np.sum(rem_band[:-1] * rem_band[1:] < 0)
    features['rem_energy'] = rem_energy
    features['rem_zero_crossings'] = rem_zero_cross
    sem_band = lowpass(sig, 0.5, fs)
    sem_variance = np.var(sem_band)
    sem_slope = np.mean(np.abs(np.diff(sem_band)))
    features['sem_variance'] = sem_variance
    features['sem_slope'] = sem_slope

    return features



def extract_emg_features(emg_signal,fs):
    features = {
        'emg_mean': np.mean(emg_signal),
        'emg_std': np.std(emg_signal),
        'emg_rms': np.sqrt(np.mean(emg_signal**2)),
    }
    features['power'] = np.mean(emg_signal**2)
    features['variance'] = np.var(emg_signal)
    f, Pxx = compute_psd_welch(emg_signal, fs=fs, nperseg=fs*2)
    band_power = np.trapezoid(Pxx[(f>=20) & (f<=40)], f[(f>=20) & (f<=40)])
    total_power = np.trapezoid(Pxx, f)
    features['hf_ratio'] = band_power / total_power if total_power > 0 else 0
    

    return features

def hjorth_complexity(epoch):

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


def visualize_feature_distributions(features_array, feature_names):

    df = pd.DataFrame(features_array, columns=feature_names)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, palette="Blues")
    plt.title("Time-Domain Feature Distribution Across Epochs")
    plt.xticks(rotation=45)
    plt.ylabel("Feature Value")
    plt.tight_layout()
    
def visualize_feature_trends(features_array, feature_names):
    plt.figure(figsize=(14, 6))
    for i, name in enumerate(feature_names):
        plt.plot(features_array[:, i], label=name)
    plt.title("Time-Domain Feature Trends Across Epochs")
    plt.xlabel("Epoch Index")
    plt.ylabel("Feature Value")
    plt.legend(ncol=4)
    plt.tight_layout()
 
def detect_blinks(data, threshold=None):
    if threshold is None:
        threshold = 2.5 * np.std(data) 
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





