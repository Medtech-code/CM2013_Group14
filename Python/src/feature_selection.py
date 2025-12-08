import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
def select_features(features, feature_names, config, labels=None, return_indices=False):
    """
    If return_indices=True, also return indices of selected features.
    """
    print(f"Selecting features for iteration {config.CURRENT_ITERATION}...")
    print(f"Input features shape: {features.shape}")

    selected_indices = None

    if features.shape[1] == 0:
        print("⚠️  WARNING: No features to select from!")
        selected_features = features
        selected_indices = np.arange(features.shape[1])

    elif config.CURRENT_ITERATION == 1:
        selected_features = features
        selected_indices = np.arange(features.shape[1])
        print("Iteration 1 Selected features shape (no selection):", selected_features.shape)

    elif config.CURRENT_ITERATION in [2, 3]:
        selector = VarianceThreshold(threshold=0.01)
        X_var = selector.fit_transform(features)
        X_corr, kept_features = drop_correlated_features(X_var, threshold=0.95)
        selected_features = X_corr
        selected_indices = kept_features
        print(f"Iteration {config.CURRENT_ITERATION} Selected features shape:", selected_features.shape)

    elif config.CURRENT_ITERATION == 4:
        if labels is None:
            raise ValueError("Labels are required for iteration 4 feature selection")

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]  # descending
        n_keep = 50
        selected_indices = indices[:n_keep]
        selected_features = features[:, selected_indices]

        # Optionally save feature names & importance plots
        selected_feature_names = [feature_names[i] for i in selected_indices]
        df_selected = pd.DataFrame(features[:, selected_indices], columns=selected_feature_names)
        df_selected.to_csv("selected_features_iter4.csv", index=False)

    if return_indices:
        return selected_features, selected_indices
    else:
        return selected_features

def drop_correlated_features(X, threshold=0.95):
    corr_matrix = np.corrcoef(X, rowvar=False)
    n_features = corr_matrix.shape[0]
    to_drop = set()

    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)

    keep_idx = [i for i in range(n_features) if i not in to_drop]
    return X[:, keep_idx], keep_idx