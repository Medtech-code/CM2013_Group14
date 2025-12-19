import numpy as np
import pandas as pd
import os

def make_inference(model, holdout_data, config):

    print("Making inference on hold-out data...")
    predictions = model.predict(holdout_data)
    return predictions

def generate_submission_file(predictions, record_numbers, epoch_numbers, config):

    print(f"Generating submission file: {config.SUBMISSION_FILE}...")
    print(f"Length of predictions: {len(predictions)}")
    print(f"Length of record_numbers: {len(record_numbers)}")
    print(f"Length of epoch_numbers: {len(epoch_numbers)}")
    submission_df = pd.DataFrame({
        'record_number': record_numbers,
        'epoch_number': epoch_numbers,
        'label': predictions
    })
    submission_df.to_csv(os.path.join(config.DATA_DIR, config.SUBMISSION_FILE), index=False)
    print("Submission file generated successfully.")
