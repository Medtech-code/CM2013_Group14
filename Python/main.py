import config
from src.data_loader import load_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import train_classifier
from src.visualization import visualize_results
from src.report import generate_report
from src.utils import save_cache, load_cache
from pathlib import Path
import os
import sys
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
 

    print("\n=== PROCESSING LOG ===")

    print(f"--- Sleep Scoring Pipeline - Iteration {config.CURRENT_ITERATION} ---")

    print("\n=== STEP 1: DATA LOADING ===")
    training_dir = config.TRAINING_DIR
    edf_file = list(Path(training_dir).glob('*.edf')) 

    all_epochs = []
    all_labels = []
    all_record_ids = []
    multi_channel_list = [] 

    for edf_path in edf_file:
        xml_path = edf_path.with_suffix('.xml')
        record_id = edf_path.stem
        
        try:
            multi_channel_data, labels, channel_info = load_training_data(str(edf_path), str(xml_path))
        
            multi_channel_list.append(multi_channel_data)
            print(f"Multi-channel data loaded:")
            print(f"  EEG: {multi_channel_data['eeg'].shape}")
            print(f"Labels shape: {labels.shape}")

            print("multi_channel_data shape:", multi_channel_data['eeg'].shape)
            print("First 10 values of first epoch:", multi_channel_data['eeg'][0, :10])
            print("Labels shape", labels.shape)
            print("First 10 labels:", labels[:10])
            
            eeg_data = multi_channel_data['eeg'][:, 0, :]  
            print("unique values:", np.unique(eeg_data)[:10])
            print(f"Using EEG channel 1 for pipeline: {eeg_data.shape}")
            all_epochs.append(eeg_data)
            all_labels.append(labels)
            all_record_ids.extend([record_id] * len(labels)) 
      

        except (ValueError, TypeError):
            eeg_data, labels = load_training_data(edf_file, xml_path)
            print(f"Single-channel data loaded: {eeg_data.shape}, Labels: {labels.shape}")


    if config.CURRENT_ITERATION == 1:
        combined_epochs = np.concatenate(all_epochs, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
    else:
        eeg_all = [mc['eeg'] for mc in multi_channel_list]
        eog_all = [mc['eog'] for mc in multi_channel_list]
        emg_all = [mc['emg'] for mc in multi_channel_list] 
        combined_epochs = {
            'eeg': np.concatenate(eeg_all, axis=0),
            'eog': np.concatenate(eog_all, axis=0),
            'emg': np.concatenate(emg_all, axis=0)  
        }
        combined_labels = np.concatenate(all_labels, axis=0)


    unique, counts = np.unique(combined_labels, return_counts=True) 
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    total = combined_labels.size

    print("\nLabel distribution across all data:")
    for stage, count in zip(stage_names, counts):
        percent = 100 * count / total
        print(f"  {stage}: {count} epochs ({percent:.1f}%)")

    if isinstance(combined_epochs, dict):
        print("\nEEG shape:", combined_epochs['eeg'].shape)
        print("EOG shape:", combined_epochs['eog'].shape)
        if 'emg' in combined_epochs:
            print("EMG shape:", combined_epochs['emg'].shape)
        num_epochs = combined_epochs['eeg'].shape[0]
    else:
        print("\nEpochs shape:", combined_epochs.shape)
        num_epochs = combined_epochs.shape[0]

    print("Labels shape:", combined_labels.shape)

    if num_epochs != combined_labels.shape[0]:
        print("Warning: Number of epochs does not match number of labels!")
    else:
        print("\nEpochs and labels are correctly aligned.")


    print("\n=== STEP 2: PREPROCESSING ===")
    preprocessed_data = None
    cache_filename_preprocess = f"preprocessed_data_iter{config.CURRENT_ITERATION}.joblib"

    print(f"Type of eeg_data: {type(combined_epochs)}")
    if isinstance(combined_epochs, dict):
        
        print(f"Keys in eeg_data: {list(combined_epochs.keys())}")
    else:
        print(f"eeg_data shape: {combined_epochs.shape}")



    if config.USE_CACHE:
        preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
        if preprocessed_data is not None:
            print("Loaded preprocessed data from cache")

    if preprocessed_data is None:
        preprocessed_data = preprocess(combined_epochs, config, channel_info)
        if config.USE_CACHE:
            save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
            print("Saved preprocessed data to cache")


    print("\n=== STEP 3: FEATURE EXTRACTION ===")
    cache_filename_features = f"features_iter{config.CURRENT_ITERATION}.joblib"
    features = None
    feature_names = None
    if config.USE_CACHE:
        cached = load_cache(cache_filename_features, config.CACHE_DIR)
        if cached is not None:
            features, feature_names = cached
            print("Loaded features + feature_names from cache")
            print(f"Features shape from cache: {features.shape}")

    if features is None:
        if config.CURRENT_ITERATION == 2:
            eeg = preprocessed_data['eeg'][:, 0, :]
            eog = preprocessed_data['eog'][:, 0, :]
            features, feature_names = extract_features([eeg, eog], config, channel_info)
        else:
            features, feature_names = extract_features(preprocessed_data, config, channel_info)

        if features is None or features.shape[1] == 0:
            print("WARNING: No features extracted! Students must implement feature extraction.")
        else:
            print(f"Extracted features shape: {features.shape}")

        if config.USE_CACHE:
            save_cache((features, feature_names), cache_filename_features, config.CACHE_DIR)
            print("Saved features + feature_names to cache")


    print("\n=== STEP 4: FEATURE SELECTION ===")

    cache_filename = f"features_selected_iter{config.CURRENT_ITERATION}.joblib"
    indices_filename = f"selected_indices_iter{config.CURRENT_ITERATION}.npy"
    selected_features = None
    selected_indices = None

    if config.USE_CACHE:
        cached = load_cache(cache_filename, config.CACHE_DIR)
        if cached is not None:
            selected_features = cached
            try:
                selected_indices = np.load(os.path.join(config.CACHE_DIR, indices_filename),allow_pickle=True)
                print("Loaded selected features and indices from cache")
            except FileNotFoundError:
                print("Loaded selected features from cache, but no indices file found")
            print(f"Selected features shape: {selected_features.shape}")

    if selected_features is None:

        if config.CURRENT_ITERATION == 1:
            selected_features = features
            selected_indices = np.arange(features.shape[1])
            print("Iteration 1: no selection")
            print(f"Selected features shape: {selected_features.shape}")

            if config.USE_CACHE:
                save_cache(selected_features, cache_filename, config.CACHE_DIR)
                np.save(os.path.join(config.CACHE_DIR, indices_filename), selected_indices)
                print(f"Saved selected features and indices to cache: {cache_filename}, {indices_filename}")

        else:
            selected_features, selected_indices = select_features(
                features, feature_names, config, labels=combined_labels, return_indices=True
            )

            if config.USE_CACHE:
                save_cache(selected_features, cache_filename, config.CACHE_DIR)
                np.save(os.path.join(config.CACHE_DIR, indices_filename), selected_indices)
                print(f"Saved selected features and indices to cache: {cache_filename}, {indices_filename}")

            print(f"Selected features shape: {selected_features.shape}")

    print("\n=== STEP 5: CLASSIFICATION ===")
    if selected_features.shape[1] > 0:
        cache_filename_model = f"model_iter{config.CURRENT_ITERATION}.joblib"
        if config.USE_CACHE:
            model = load_cache(cache_filename_model, config.CACHE_DIR)
            if model is not None:
                print("Loaded model from cache")
            else:
                model = train_classifier(selected_features, combined_labels, all_record_ids, config)
                print(f"Trained {config.CLASSIFIER_TYPE} classifier")
                save_cache(model,cache_filename_model, config.CACHE_DIR)
    else:
        print("WARNING: Cannot train classifier - no features available!")
        print("Students must implement feature extraction first.")
        model = None

    print("\n=== STEP 6: VISUALIZATION ===")
    if model is not None:
        print()
    else:
        print("Skipping visualization - no trained model")
    print("\n=== STEP 7: PROCESSING LOG & REPORT GENERATION ===") 
     
    if model is not None:
        generate_report(model, selected_features, combined_labels, config, txt_filename=None)

    else:
        print("Skipping report - no trained model")

    print("\n" + "="*50)
    print("PIPELINE FINISHED")
    print("="*50)

if __name__ == "__main__":
    main()