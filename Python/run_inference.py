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


def process_holdout_file(file_path, model, scaler,config):

    record_id = Path(file_path).stem
    print(f"\n  Processing edf file {record_id}...")
    try:
        holdout_data, record_info, channel_info = load_holdout_data(file_path)
    except Exception as e:
        print(f"error! Failed to load {record_id}: {e}")
        return None, None
    preprocessed_holdout_data = None
    cache_filename_preprocess_holdout = f"preprocessed_holdout_data_iter{config.CURRENT_ITERATION}.joblib"

    if preprocessed_holdout_data is None:
        try:
            if config.CURRENT_ITERATION==1:
                preprocessed_holdout_data = preprocess(holdout_data['eeg'][:,0,:], config, channel_info)
            else:
                preprocessed_holdout_data=preprocess(holdout_data,config,channel_info)
        except Exception as e:
            print(f"error! Failed preprocessing for {record_id}: {e}")
            return None, None   
        if config.USE_CACHE:
            save_cache(preprocessed_holdout_data, cache_filename_preprocess_holdout, config.CACHE_DIR)
    print(f"******************************2preprocessed_holdout_data: {preprocessed_holdout_data}")
        
    holdout_features = None
    cache_filename_features_holdout = f"features_holdout_iter{config.CURRENT_ITERATION}.joblib"

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
            return None, None        
        if config.USE_CACHE:
            save_cache(holdout_features, cache_filename_features_holdout, config.CACHE_DIR)
    print(f"******************************holdout_features: {holdout_features}")

    selected_holdout_features=None
    cache_filename_selected_features_holdout = f"features_selectedholdout_iter{config.CURRENT_ITERATION}.joblib"

    if selected_holdout_features is None:
        try:
            indices_filename = f"selected_indices_iter{config.CURRENT_ITERATION}.npy"
            indices_path = os.path.join(config.CACHE_DIR, indices_filename)
            print(f"Loading selected indices from: {indices_path}")
            selected_indices = np.load(indices_path)
            selected_holdout_features = holdout_features[:, selected_indices]
                
        except Exception as e:
            print(f"error! Failed feature extraction for {record_id}: {e}")
            return None, None        
        if config.USE_CACHE:
            save_cache(selected_holdout_features, cache_filename_selected_features_holdout, config.CACHE_DIR)

    print(f"******************************selected_holdout_features: {selected_holdout_features}")      

    try:
        prediction = make_inference(model, selected_holdout_features, config)
        print(f"Prediction for {record_id} successful: {prediction}")
        print("DEBUG CHECHILIST----------------")
        print(f"NaN in features: {np.isnan(selected_holdout_features).any()}")
        print(f"Inf in features: {np.isinf(selected_holdout_features).any()}")
        print(f"Feature min: {selected_holdout_features.min()}, max: {selected_holdout_features.max()}")

        unique, counts = np.unique(prediction, return_counts=True)
        for stage, count in zip(unique, counts):
            print(f"Stage {stage}: {count} epochs ({100*count/len(prediction):.1f}%)")
        print("DEBUG CHECHILIST END----------------")
        
        return prediction, record_info
    except Exception as e:
        print(f"error! Failed prediction for {record_id}: {e}")
        return None, None
        
        
def run_inference():
    print(f"--- Sleep Scoring Inference - Iteration {config.CURRENT_ITERATION} ---")
    model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
    model = load_cache(model_filename, config.CACHE_DIR)
    if model is None:
        print("Error: Trained model not found. Please run main.py first to train a model.")
        return
    
    scaler = None
    holdout_dir_path = Path(config.HOLDOUT_DIR)
    holdout_files = sorted(holdout_dir_path.glob("*.edf"))
    if holdout_files is None:
        print("ERROR: Holdout_files failed to load")
        return
    predictions = []
    epoch_numbers = []
    record_numbers = []
    for file in holdout_files:
        prediction, record_info = process_holdout_file(str(file), model, scaler, config)
        if prediction.size > 0 and record_info:
            actual_n_epochs = len(prediction)
            record_id = record_info['record_id']
            
            predictions.extend(prediction)
            
            record_int = int(record_id.replace('H', '')) if record_id.startswith('H') else 0
            record_numbers.extend([record_int] * actual_n_epochs)
            epoch_numbers.extend(range(1, actual_n_epochs + 1))

    predictions = np.array(predictions)

    
    generate_submission_file(predictions, record_numbers, epoch_numbers, config)

    print("--- Inference Finished ---")

if __name__ == "__main__":
    run_inference()