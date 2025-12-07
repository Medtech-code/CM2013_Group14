import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
def select_features(features, labels,feature_names, config):
    """
    STUDENT IMPLEMENTATION AREA: Select most relevant features.

    Feature selection becomes important in later iterations to:
    1. Reduce overfitting
    2. Improve computation speed
    3. Focus on most discriminative features
    4. Handle curse of dimensionality

    Suggested approaches for students to implement:
    - Statistical tests (ANOVA F-test, chi-square)
    - Mutual information
    - Correlation-based selection
    - Recursive feature elimination
    - L1 regularization (LASSO)
    - Tree-based feature importance

    Args:
        features (np.ndarray): The input features (n_samples, n_features).
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.

    Returns:
        np.ndarray: The selected features (n_samples, n_selected_features).
    """
    print(f"Selecting features for iteration {config.CURRENT_ITERATION}...")
    print(f"Input features shape: {features.shape}")

    if features.shape[1] == 0:
        print("⚠️  WARNING: No features to select from!")
        return features
    
    if config.CURRENT_ITERATION == 1:
        selected_features = features
        print("Iteration 1 Selected features shape (no selection):", selected_features.shape)
        
    elif config.CURRENT_ITERATION == 2:
        # Early iterations: Use all available features
        selector = VarianceThreshold(threshold=0.01)  
        X_var = selector.fit_transform(features)
        corr_matrix = np.corrcoef(X_var, rowvar=False)
        selector = SelectKBest(score_func=f_classif, k=50)
        X_corr, kept_features = drop_correlated_features(X_var, threshold=0.95)
        X_selected = selector.fit_transform(X_corr, labels)
        print("Iteration 2 Selected features shape:", selected_features.shape)

        selected_features = X_selected

    elif config.CURRENT_ITERATION == 3:

        selector = VarianceThreshold(threshold=0.01)  
        X_var = selector.fit_transform(features)
        corr_matrix = np.corrcoef(X_var, rowvar=False)
        selector = SelectKBest(score_func=f_classif, k=50)
        X_corr, kept_features = drop_correlated_features(X_var, threshold=0.95)
        X_selected = selector.fit_transform(X_corr, labels)
        print("Iteration 3 Selected features shape:", selected_features.shape)

        selected_features = X_selected

    
    elif config.CURRENT_ITERATION == 4:

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
        indices = np.argsort(importances)[::-1]  # sorted biggest → smallest
        print("\n===== TOP 20 MOST IMPORTANT FEATURES =====")

        for i in range(20):
            idx = indices[i]
            print(f"{i+1:2d}. {feature_names[idx]:45s} : {importances[idx]:.5f}")
        n_keep = 50   # change if needed
        selected_indices = indices[:n_keep]
        selected_feature_names = [feature_names[i] for i in selected_indices]

        print(f"\nSelected top {n_keep} features stored in selected_feature_names")
        df_selected = pd.DataFrame(features[:, selected_indices],
                                columns=selected_feature_names)
        df_selected.to_csv("selected_features_iter4.csv", index=False)
        print("\nSaved: selected_features_iter4.csv")


        plt.figure(figsize=(14, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xlabel("Features (sorted)")
        plt.ylabel("Importance")
        plt.title("Random Forest Feature Importance")
        plt.axvline(x=n_keep, linestyle='--', color='red', label=f"Top {n_keep}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("feature_importance_iter4.png", dpi=150)
        plt.show()

        print("Saved: feature_importance_iter4.png")
        selected_features = features[:, selected_indices]  
        print("Iteration 4 Selected features shape:", selected_features.shape)


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