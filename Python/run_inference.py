from pathlib import Path
import numpy as np
import config
from src.data_loader import load_holdout_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.inference import make_inference, generate_submission_file
from src.utils import save_cache, load_cache
import os
import joblib
import glob


def process_holdout_file(file_path, model, config):
    """
    Processes one holdout EDF file.

    Args:
        file_path (str): EDF file path
        model (_type_): Model of classifier
        config (_type_): Config file
    Returns:
        tuple: (prediction_data, record_info) where:
            - prediction_data (np.ndarray): Predictions of sleep stage for each epoch
            - record_info (dict): Metadata including record_id, n_epochs, channels
    """
    
    record_id = Path(file_path).stem
    print(f"\n  Processing edf file {record_id}...")
    
    # 1. Load Hold-out Data
    try:
        holdout_eeg_data, record_info, channel_info = load_holdout_data(file_path)
    except Exception as e:
        print(f"error! Failed to load {record_id}: {e}")
        return
    
    # 2. Preprocessing (using the same logic as training)
    preprocessed_holdout_data = None
    cache_filename_preprocess_holdout = f"preprocessed_holdout_data_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        preprocessed_holdout_data = load_cache(cache_filename_preprocess_holdout, config.CACHE_DIR)

    if preprocessed_holdout_data is None:
        try:
            preprocessed_holdout_data = preprocess(holdout_eeg_data, config, channel_info)
        except Exception as e:
            print(f"error! Failed preprocessing for {record_id}: {e}")
            return None        
        if config.USE_CACHE:
            save_cache(preprocessed_holdout_data, cache_filename_preprocess_holdout, config.CACHE_DIR)
            

    # 3. Feature Extraction (using the same logic as training)
    holdout_features = None
    cache_filename_features_holdout = f"features_holdout_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        holdout_features = load_cache(cache_filename_features_holdout, config.CACHE_DIR)
        print(holdout_features)
    if holdout_features is None:
        try:
            if config.CURRENT_ITERATION==2:
                preprocessed_holdout_data_eeg=preprocessed_holdout_data['eeg'][:, 0, :]
                preprocessed_holdout_data_eog=preprocessed_holdout_data['eog'][:,0,:]
                holdout_features,feature_names = extract_features([preprocessed_holdout_data_eeg,preprocessed_holdout_data_eog], config, channel_info)

            else:

                holdout_features,feature_names = extract_features(preprocessed_holdout_data, config, channel_info)
    

        except Exception as e:
            print(f"error! Failed feature extraction for {record_id}: {e}")
            return None        
        if config.USE_CACHE:
            save_cache(holdout_features, cache_filename_features_holdout, config.CACHE_DIR)

    # Feature selection
    selected_holdout_features=None
    cache_filename_selectedfeatures_holdout = f"features_selectedholdout_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        holdout_selected_features = load_cache(cache_filename_selectedfeatures_holdout, config.CACHE_DIR)

    if selected_holdout_features is None:
        try:
            selected_indices = np.load(f"cache\selected_indices_iter{config.CURRENT_ITERATION}.npy")
            holdout_selected_features = holdout_features[:, selected_indices]
        except Exception as e:
            print(f"error! Failed feature extraction for {record_id}: {e}")
            return None        
        if config.USE_CACHE:
            save_cache(holdout_selected_features, cache_filename_selectedfeatures_holdout, config.CACHE_DIR)

    
    # 4. Make Inference
    try:
        prediction = make_inference(model, holdout_selected_features, config)
        print(f"Prediction for {record_id} successful")
        return prediction, record_info
    except Exception as e:
        print(f"error! Failed prediction for {record_id}: {e}")
        return None
        
        
def run_inference():
    print(f"--- Sleep Scoring Inference - Iteration {config.CURRENT_ITERATION} ---")

    # Load the trained model (assuming it was saved during training)
    model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
    model = load_cache(model_filename, config.CACHE_DIR)
    if model is None:
        print("Error: Trained model not found. Please run main.py first to train a model.")
        return

    # 1. Load Hold-out Data files
    #           -For iterating through files.
    holdout_dir_path = Path(config.HOLDOUT_DIR)
    holdout_files = sorted(holdout_dir_path.glob("*.edf"))
    if holdout_files is None:
        print("ERROR: Holdout_files failed to load")
        return
    predictions = []
    epoch_numbers = []
    record_numbers = []
    # 2. Preprocessing each file
    
    for file in holdout_files:
        prediction, record_info = process_holdout_file(str(file), model, config)
        if prediction and record_info:

            n_epochs = record_info['n_epochs']
            record_id = record_info['record_id']
            
            predictions.extend(prediction)
            
            record_int = int(record_id.replace('H', '')) if record_id.startswith('H') else 0
            record_numbers.extend([record_int] * n_epochs)
            epoch_numbers.extend(range(1, n_epochs + 1))

    predictions = np.array(predictions)


    # 5. Generate Submission File
    generate_submission_file(predictions, record_numbers, epoch_numbers, config)

    print("--- Inference Finished ---")

if __name__ == "__main__":
    run_inference()