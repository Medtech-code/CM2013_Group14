import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import config
import os
output_dir = f"result/iteration_{config.CURRENT_ITERATION}"
os.makedirs(output_dir, exist_ok=True)
def select_features(features, feature_names, config, labels=None, return_indices=False):

    print(f"Selecting features for iteration {config.CURRENT_ITERATION}...")
    print(f"Input features shape: {features.shape}")

    selected_indices = None
    selected_feature_names = feature_names

    if features.shape[1] == 0:
        selected_features = features

    elif config.CURRENT_ITERATION == 1:
        selected_features = features
        selected_indices = np.arange(features.shape[1])

    elif config.CURRENT_ITERATION == 2:
        selector = VarianceThreshold(threshold=0.01)
        X_var = selector.fit_transform(features)
        X_corr, kept_features = drop_correlated_features(X_var, threshold=0.95)
        feature_variances = X_corr.var(axis=0)
        if len(feature_variances) > 50:
            top_indices = np.argsort(feature_variances)[::-1][:50] 
            selected_features = X_corr[:, top_indices]
            selected_indices = [kept_features[i] for i in top_indices]
        else:
            selected_features = X_corr
            selected_indices = kept_features
        
        selected_feature_names = [feature_names[i] for i in selected_indices]

    elif config.CURRENT_ITERATION in [3,4]:
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
        indices = np.argsort(importances)[::-1]
        print("Feature Name\tImportance Score")
        for idx in indices:
            print(f"{feature_names[idx]}\t{importances[idx]:.4f}")

        n_keep = 50
        selected_indices = indices[:n_keep]
        selected_features = features[:, selected_indices]
        selected_feature_names = [feature_names[i] for i in selected_indices]

        print(f"Iteration 4 Selected features shape: {selected_features.shape}")

    try:
        df_selected = pd.DataFrame(selected_features, columns=selected_feature_names)
        csv_filename = os.path.join(
            output_dir,
            f"selected_features_iter{config.CURRENT_ITERATION}.csv"
        )

        df_selected.to_csv(csv_filename, index=False)
        print(f"Saved selected features CSV: {csv_filename}")
        print(f"Saved selected features CSV: {csv_filename}")
    except Exception as e:
                print(f"Failed to save CSV: {e}")

    if return_indices:
        return selected_features, selected_indices
    else:
        return selected_features

def drop_correlated_features(X, threshold=0.9):
    corr_matrix = np.corrcoef(X, rowvar=False)
    n_features = corr_matrix.shape[0]
    to_drop = set()

    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)

    keep_idx = [i for i in range(n_features) if i not in to_drop]
    return X[:, keep_idx], keep_idx
