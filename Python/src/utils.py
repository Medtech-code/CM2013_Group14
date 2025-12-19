import os
import joblib

def save_cache(data, filename, cache_dir):

    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)
    joblib.dump(data, filepath)
    print(f"Data cached to {filepath}")

def load_cache(filename, cache_dir):

    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        print(f"Loading data from cache: {filepath}")
        return joblib.load(filepath)
    print(f"Cache file not found: {filepath}")
    return None
