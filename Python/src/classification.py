import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from src.utils import save_cache, load_cache
from sklearn.base import clone



def train_classifier(features, labels, all_record_ids, config):
    """
    STUDENT IMPLEMENTATION AREA: Train classifier based on iteration.

    This function provides a basic framework but students should enhance it:

    1. Implement proper cross-validation (not just train/test split)
    2. Address class imbalance in sleep stage data
    3. Tune hyperparameters for each classifier
    4. Add more sophisticated evaluation metrics
    5. Consider ensemble methods in later iterations

    Args:
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        all_record_ids (array): Record id array, e.g. ['R1', 'R1', ..., 'R2', 'R2', ..., 'R10', 'R10', ...]
        config (module): The configuration module.

    Returns:
        object: The trained classifier.
    """
    print(f"Training {config.CLASSIFIER_TYPE} classifier...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Basic validation
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("No features available for training!")

    # BASIC train/test split - students should implement cross-validation
    # TODO: Students should implement k-fold cross-validation for more robust evaluation
    # Use stratified split for realistic sleep data distribution
    # Sleep stages are naturally imbalanced (more N2, less N1/REM)
    '''
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print("Using stratified train/test split to maintain class balance")
    except ValueError as e:
        # Fallback for edge cases (very small datasets)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        print(f"Using non-stratified split: {e}")
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")'''

    # TODO: Students should address class imbalance in sleep data:
    # - Sleep stages are not equally distributed
    # - Consider SMOTE, class weights, or other techniques
    #smote = SMOTE(random_state=42)
    #X_train, y_train = smote.fit_resample(X_train, y_train)



    logo = LeaveOneGroupOut()
    
    # Select classifier based on iteration (using config parameters)
    if config.CURRENT_ITERATION == 1:
        # Iteration 1: Simple k-NN
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS))
        ])
        model = pipeline
        print(f"Using k-NN with k={config.KNN_N_NEIGHBORS}")

    elif config.CURRENT_ITERATION == 2:
        # Iteration 2: SVM
        # TODO: Students should tune hyperparameters (C, kernel, gamma)
        '''model = SVC(
            C=getattr(config, 'SVM_C', 1.0),
            kernel=getattr(config, 'SVM_KERNEL', 'rbf'),
            random_state=42
        )
        print(f"Using SVM with C={model.C}, kernel={model.kernel}")'''
        # 1. Define the Pipeline (Scaler -> SVM)
        pipeline = ImbPipeline([
            #('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42))
        ])
        
        # 2. Define the Parameter Grid (Start small due to speed warning)
        # Note: 'svm__C' and 'svm__gamma' link the parameter to the 'svm' step in the pipeline.
        param_grid = {
            'svm__C': [0.1, 0.12, 0.14],
            'svm__kernel': ['linear'],  # Linear performed best, focus on it
            'svm__gamma': ['scale'],  # Keep 'scale' since it worked
            'svm__class_weight': [None, 'balanced']  # Test balanced vs None
        }
        # 3. Define the Cross-Validation Strategy (3-fold subject-wise)
        
        # Use 3 or 5 splits as recommended to save time
        group_kfold = GroupKFold(n_splits=5) 

        # 4. Initialize GridSearchCV
        # Pass the pipeline, the parameter grid, and the GroupKFold strategy
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=group_kfold.split(features, labels, groups=all_record_ids), # Pass the split iterator
            scoring='accuracy', # Use 'accuracy' or 'f1_macro' (recommended for imbalanced data)
            n_jobs=-1,          # Use all available cores for speed
            verbose=2
        )

        print("Starting SVM Hyperparameter Tuning with 3-Fold GroupKFold...")
        # 5. Run the Grid Search
        # The 'groups' parameter here is essential for GroupKFold/LOSO
        grid_search.fit(features, labels, groups=all_record_ids) 

        # 6. Output Results
        print("\n" + "="*50)
        print(f"✅ Best Hyperparameters: {grid_search.best_params_}")
        print(f"🏆 Best Mean Cross-Validation Score (Accuracy or F1): {grid_search.best_score_:.3f}")
        print("="*50)

        # The best model (Pipeline object) is now stored and ready for evaluation
        model = grid_search.best_estimator_
        print(f"Using SVM with params:{grid_search.best_estimator_}")

    elif config.CURRENT_ITERATION >= 3:
        # Iteration 3+: Random Forest
        # TODO: Students should tune hyperparameters (n_estimators, max_depth, etc.)
        
        base_pipe_rf = ImbPipeline([
        #('smote', SMOTE(random_state=42)),
        #('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=30,
            min_samples_leaf=4,
            min_samples_split=2,
            n_estimators=200,
        )),
        ]) 
        
        param_grid = {
            'classifier__n_estimators': [200, 400],
            'classifier__max_depth': [20, 30, 50],
            'classifier__min_samples_split': [2, 7],
            'classifier__min_samples_leaf': [1, 4],
            'classifier__class_weight': ['balanced']
        }
        group_kfold = GroupKFold(n_splits=5) 
        
        print("Starting RandomForest hyperparameter tuning with LOSO GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=base_pipe_rf,
            param_grid=param_grid,
            cv=logo.split(features, labels, groups=all_record_ids),
            scoring='f1_macro',     # good for imbalanced multi-class
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(features, labels, groups=all_record_ids)

        print("\n" + "="*60)
        print(f"✅ Best RF Hyperparameters: {grid_search.best_params_}")
        print(f"🏆 Best Mean LOSO CV F1-macro: {grid_search.best_score_:.3f}")
        print("="*60)

        # Best pipeline (SMOTE + scaler + RF with tuned params)
        model = grid_search.best_estimator_
        print(f"Using Random Forest with params:{grid_search.best_estimator_}")
        

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    ''' older simpler 
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)

    # Comprehensive evaluation with detailed performance metrics
    y_pred = model.predict(X_test)
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy for : {overall_accuracy:.3f}")

    # Calculate and display detailed performance metrics
    print_performance_metrics(y_test, y_pred)           '''

    # TODO: Students should add more advanced metrics:
    # - Cohen's kappa (important for sleep scoring)
    # - ROC-AUC for each class
    # - Cross-validation scores
    # - Feature importance analysis
    print("\nTODO: Students should add Cohen's kappa and ROC-AUC metrics")

    # Assuming you tracked record_ids when loading data
    # record_ids is array like ['R1', 'R1', ..., 'R2', 'R2', ..., 'R10', 'R10', ...]
    # Create LOSO cross-validation split
    
    loso_results = []
    all_y_test = [] 
    all_y_pred = []

    #smote resampling also for cross validation
    #smote = SMOTE(random_state=42)
    all_record_ids = np.array(all_record_ids)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(features, labels, groups=all_record_ids)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        '''X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # IMPORTANT: Feature scaling within each fold (no data leakage!)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)'''
        

        # Which subject is held out in this fold?   
        test_subject = np.unique(all_record_ids[test_idx])[0]
        print(f"Fold {fold_idx+1}/10: Training on 9 subjects, testing on {test_subject}")

        # Train classifier on 9 subjects
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        #model.fit(X_train, y_train)

        # Predict on held-out subject
        y_pred = fold_model.predict(X_test)
        
        # Aggregate for final aggregated Confusion Matrix
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        # Calculate metrics for this subject
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        # Per-class F1 scores
        f1_per_class = f1_score(y_test, y_pred, average=None)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        loso_results.append({
            'subject': test_subject,
            'accuracy': accuracy,
            'kappa': kappa,
            'f1_macro': f1_macro
        })

        print(f"  {test_subject}: Accuracy={accuracy:.1%}, Kappa={kappa:.3f}, F1-macro={f1_macro:.3f}")
        
        print(f"        ------------SLEEP METRICS------------")
        
        # Compare ground truth vs predictions
        true_metrics = calculate_sleep_metrics(y_test)
        pred_metrics = calculate_sleep_metrics(y_pred)

        # Report differences
        for metric_name in true_metrics:
            true_val = true_metrics[metric_name]
            pred_val = pred_metrics[metric_name]
            error = abs(pred_val - true_val)
            print(f"{metric_name}: True={true_val:.1f}, Pred={pred_val:.1f}, Error={error:.1f}")    
            

    # Report mean ± std across all 10 subjects
    mean_acc = np.mean([r['accuracy'] for r in loso_results])
    std_acc = np.std([r['accuracy'] for r in loso_results])
    mean_kappa = np.mean([r['kappa'] for r in loso_results])
    std_kappa = np.std([r['kappa'] for r in loso_results])

    print("\n" + "="*60)
    print(f"LOSO Cross-Validation Results (10 subjects):")
    print(f"  Accuracy = {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"  Kappa    = {mean_kappa:.3f} ± {std_kappa:.3f}")
    print("="*60)
    
    # Confusion matrix aggregated across all LOSO folds
    print_performance_metrics(np.array(all_y_test), np.array(all_y_pred))

    # Show per-subject variability
    print("\nPer-Subject Performance:")
    for r in sorted(loso_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"  {r['subject']}: {r['accuracy']:.1%} (kappa={r['kappa']:.3f})")
        
    ''' 
    final_model_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)), # Resample the whole dataset (if desired for final model)
    ('scaler', StandardScaler()),
    ('classifier', model) # Use the same classifier you defined earlier (e.g., k-NN)
    ])'''

    print("\n" + "="*60)
    print("Training FINAL MODEL on ALL 10 Subjects' Data...")
    final_model_pipeline = clone(model)
    final_model_pipeline.fit(features, labels)
    
    print("Final model trained successfully.")
    print("="*60)
    
    '''
    if hasattr(final_model_pipeline, "named_steps") and 'scaler' in final_model_pipeline.named_steps:
        final_scaler = final_model_pipeline.named_steps['scaler']
        scaler_filename = f"scaler_iter{config.CURRENT_ITERATION}.joblib"
        save_cache(final_scaler, scaler_filename, config.CACHE_DIR)
        print(f"✅ Saved final scaler to {config.CACHE_DIR}/{scaler_filename}") '''
    
    
    # 2. Save the FITTED scaler object
    #scaler_filename = f"scaler_iter{config.CURRENT_ITERATION}.joblib"
    # Assuming you have a save_cache function that uses joblib:
    #save_cache(final_scalar, scaler_filename, config.CACHE_DIR)
    #print(f"✅ Saved final scaler to {config.CACHE_DIR}/{scaler_filename}") 
    
    
    print("\nSanity check: prediction distribution on TRAINING data with final model")
    train_pred = final_model_pipeline.predict(features)
    unique, counts = np.unique(train_pred, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Predicted label {u}: {c} samples ({100*c/len(train_pred):.1f}%)")
            
    return final_model_pipeline


def print_performance_metrics(y_true, y_pred):
    """
    Print comprehensive performance metrics for sleep stage classification.

    Includes accuracy, sensitivity (recall), specificity, and F1-score for each sleep stage.
    """

    # Sleep stage labels and names (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_labels = list(range(5))

    print("\n" + "="*70)
    print("SLEEP STAGE CLASSIFICATION PERFORMANCE METRICS")
    print("="*70)

    # Overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Macro F1-Score: {macro_f1:.3f}")
    print(f"Weighted F1-Score: {weighted_f1:.3f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=stage_labels)

    # Create a formatted confusion matrix
    cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
    print(cm_df.to_string())

    # Per-class metrics
    print("\nPer-Class Performance Metrics:")
    print("-" * 70)
    print(f"{'Stage':<8} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1-Score':<10}")
    print("-" * 70)

    # Calculate metrics for each sleep stage
    for i, stage_name in enumerate(stage_names):
        if i in y_true:  # Only calculate if stage is present in test set
            # Per-class accuracy (percentage of this class correctly classified)
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((y_pred == i) & (y_true == i)) / np.sum(class_mask)
            else:
                class_accuracy = 0.0

            # Sensitivity (Recall) - True Positive Rate
            sensitivity = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

            # Specificity - True Negative Rate
            tn = np.sum((y_true != i) & (y_pred != i))
            fp = np.sum((y_true != i) & (y_pred == i))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # F1-Score
            f1 = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

            print(f"{stage_name:<8} {class_accuracy:<10.3f} {sensitivity:<12.3f} {specificity:<12.3f} {f1:<10.3f}")
        else:
            print(f"{stage_name:<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print("-" * 70)

    # Class distribution in test set
    print("\nClass Distribution in Test Set:")
    unique, counts = np.unique(y_true, return_counts=True)
    total_samples = len(y_true)

    for stage_idx, count in zip(unique, counts):
        stage_name = stage_names[stage_idx]
        percentage = count / total_samples * 100
        print(f"{stage_name}: {count} samples ({percentage:.1f}%)")

    # Sleep scoring specific notes
    print("\nNotes for Sleep Scoring:")
    print("- Sensitivity = Recall = True Positive Rate (correctly identified stages)")
    print("- Specificity = True Negative Rate (correctly rejected stages)")
    print("- Sleep stage imbalance is natural (more N2, less N1/REM)")
    print("- Consider Cohen's kappa for chance-corrected agreement")
    print("- Clinical focus: High sensitivity for REM and N3 stages")
    
def calculate_sleep_metrics(labels, epoch_duration=30):
    """
    Calculate sleep architecture metrics from epoch labels.

    Args:
        labels: array of sleep stage labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        epoch_duration: seconds per epoch (default 30)

    Returns:
        metrics: dict of sleep architecture values
    """
    
    print("-----------------------LABELS DEBUG PRINT----------")
    ctr_SOL, ctr_REM, ctr_TST, ctr_TIB, waso_count, awakenings = 0, 0, 0, 0, 0, 0
    stage_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    prev_label = None
    found_sleep_onset, found_REM = False, False
    for l in labels:
        ctr_TIB += 1
        
        # Count stages
        stage_counts[l] += 1

        #print(l, end=',')  #print all stage labels
        if found_sleep_onset is False:
            if l == 0:
                ctr_SOL += 1
            else: # found first sleep epoch
                found_sleep_onset = True
        
        if found_sleep_onset and found_REM is not True:
            if l == 4 :
                found_REM = True
            elif l != 0:
                ctr_REM += 1
        
        if found_sleep_onset and l == 0:
            waso_count += 1
            
        if l != 0:
            ctr_TST += 1
        
        if ctr_TIB > 1:
            if l == 0 and prev_label != 0:
                awakenings += 1
        prev_label = l # Update for the next epoch chec
            
    print()
    # Students must implement based on definitions above
    metrics = {}

    # 1. Find sleep onset (first non-wake epoch)
    sleep_onset = (epoch_duration * ctr_SOL) /60
    print(f"Sleep onset: {sleep_onset} min        normal 10-20min")
    
    REM_latency = (epoch_duration * ctr_REM) /60
    print(f"REM latency: {REM_latency} min      normal 70-120min")
    
    total_sleep_time = (epoch_duration * ctr_TST) /60
    total_sleep_time_h= total_sleep_time / 60
    print(f"Total sleep time: {total_sleep_time_h} h      normal 6-8h")

    ctr_TIB = len(labels)
    time_in_bed = (epoch_duration * ctr_TIB) /60
    sleep_efficiency = (total_sleep_time  / time_in_bed) *100
    print(f"Sleep efficiency: {sleep_efficiency}%        normal >85%")
    
    # WASO (Wake After Sleep Onset)    
    WASO = (epoch_duration * waso_count) / 60  # minutes
        
    total_epochs = len(labels)
    stage_percentages = {
        'Wake': (stage_counts[0] / total_epochs) * 100,
        'N1': (stage_counts[1] / total_epochs) * 100,
        'N2': (stage_counts[2] / total_epochs) * 100,
        'N3': (stage_counts[3] / total_epochs) * 100,
        'REM': (stage_counts[4] / total_epochs) * 100
    }
    
    metrics = {
        'Sleep_Onset': float(sleep_onset),  # minutes
        'REM_latency': float(REM_latency),  # minutes
        'Total_Sleep_Time': float(total_sleep_time_h),  # hours
        #'time_in_bed': float(time_in_bed),  # hours
        'Sleep_Efficiency': float(sleep_efficiency),  # percentage
        'WASO': float(WASO),  # minutes
        'Awakenings': int(awakenings),
        #'Wake_count': int(stage_counts[0]),
        #'N1_count': int(stage_counts[1]),
        #'N2_count': int(stage_counts[2]),
        #'N3_count': int(stage_counts[3]),
        #'REM_count': int(stage_counts[4]),
        'Wake': float(stage_percentages['Wake']),
        'N1': float(stage_percentages['N1']),
        'N2': float(stage_percentages['N2']),
        'N3': float(stage_percentages['N3']),
        'REM': float(stage_percentages['REM'])
    }

    return metrics
