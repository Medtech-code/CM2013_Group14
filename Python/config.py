CURRENT_ITERATION = 4
USE_CACHE = True 
import os
DATA_DIR = '../data/'
TRAINING_DIR = f'{DATA_DIR}training/'
HOLDOUT_DIR = f'{DATA_DIR}holdout/'
SAMPLE_DIR = f'{DATA_DIR}sample/'
CACHE_DIR = 'cache/'

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}\nPlease ensure you are running from the correct directory.")
if not os.path.exists(CACHE_DIR):
    print(f"Creating cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

LOW_PASS_FILTER_FREQ = 40  
HIGH_PASS_FILTER_FREQ = 0.1 

if CURRENT_ITERATION == 1:
    CLASSIFIER_TYPE = 'knn'
    KNN_N_NEIGHBORS = 5
    METHOD = CURRENT_ITERATION
elif CURRENT_ITERATION == 2:
    CLASSIFIER_TYPE = 'svm'
    SVM_C = 1.0
    SVM_KERNEL = 'rbf'
    METHOD = 'welch_wavelet'  
elif CURRENT_ITERATION == 3:
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
elif CURRENT_ITERATION == 4:
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 5
else:
    raise ValueError(f"Invalid CURRENT_ITERATION: {CURRENT_ITERATION}. Must be 1-4.")

SUBMISSION_FILE = 'submission.csv'