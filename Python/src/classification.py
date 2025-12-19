import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import recall_score, f1_score
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
from src.utils import save_cache
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
def train_classifier(features, labels, all_record_ids, config):
    print(f"Training {config.CLASSIFIER_TYPE} classifier...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("No features available for training!")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print("Using stratified train/test split to maintain class balance")
    except ValueError as e:

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        print(f"Using non-stratified split: {e}")
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")


    if config.CURRENT_ITERATION == 1:
        model = KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)
        print(f"Using k-NN with k={config.KNN_N_NEIGHBORS}")

    elif config.CURRENT_ITERATION == 2:
        '''model = SVC(
            C=getattr(config, 'SVM_C', 1.0),
            kernel=getattr(config, 'SVM_KERNEL', 'rbf'),
            random_state=42
        )
        print(f"Using SVM with C={model.C}, kernel={model.kernel}")'''
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', random_state=42))
        ])
        
        param_grid = {
            'svm__C': [0.1, 0.12, 0.14], 
            'svm__kernel': ['linear'],  
            'svm__gamma': ['scale'],  
            'svm__class_weight': [None, 'balanced']  
        }
        group_kfold = GroupKFold(n_splits=5) 
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=group_kfold.split(features, labels, groups=all_record_ids), 
            scoring='accuracy', 
            n_jobs=-1,       
            verbose=2
        )

        print("Starting SVM Hyperparameter Tuning with 3-Fold GroupKFold...")
        grid_search.fit(features, labels, groups=all_record_ids) 
        print("\n" + "="*50)
        print(f"Best Hyperparameters: {grid_search.best_params_}")
        print(f"Best Mean Cross-Validation Score (Accuracy or F1): {grid_search.best_score_:.3f}")
        print("="*50)
        model = grid_search.best_estimator_
        print(f"Using SVM with params:{grid_search.best_estimator_}")

    elif config.CURRENT_ITERATION >= 3:
        model = RandomForestClassifier(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 100),
            max_depth=getattr(config, 'RF_MAX_DEPTH', None),
            min_samples_split=getattr(config, 'RF_MIN_SAMPLES_SPLIT', 2),
            random_state=42,
            n_jobs=-1  
        )
        print(f"Using Random Forest with {model.n_estimators} trees")

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    logo = LeaveOneGroupOut()

    loso_results = []
    all_y_test = [] 
    all_y_pred = []
    all_arch_true = []
    all_arch_pred = []
    
    smote = SMOTE(random_state=42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(features, labels, groups=all_record_ids)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train, y_train = smote.fit_resample(X_train, y_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        all_record_ids = np.array(all_record_ids)
        test_subject = np.unique(all_record_ids[test_idx])[0]
        print(f"Fold {fold_idx+1}/10: Training on 9 subjects, testing on {test_subject}")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
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
        
        true_metrics = calculate_sleep_metrics(y_test)
        pred_metrics = calculate_sleep_metrics(y_pred)
        all_arch_true.append(true_metrics)
        all_arch_pred.append(pred_metrics)

        if fold_idx == 0:
            example_subject_id = test_subject
            example_y_true = y_test.copy()
            example_y_pred = y_pred.copy()
            example_true_metrics = true_metrics
            example_pred_metrics = pred_metrics

        for metric_name in true_metrics:
            true_val = true_metrics[metric_name]
            pred_val = pred_metrics[metric_name]
            error = abs(pred_val - true_val)
            print(f"{metric_name}: True={true_val:.1f}, Pred={pred_val:.1f}, Error={error:.1f}")   


    arch_metric_names = list(all_arch_true[0].keys())
    iter_dir = os.path.join("result", f"iteration_{config.CURRENT_ITERATION}")
    os.makedirs(iter_dir, exist_ok=True)

    for m in arch_metric_names:
        true_vals = np.array([t[m] for t in all_arch_true])
        pred_vals = np.array([p[m] for p in all_arch_pred])

        mean_vals = (true_vals + pred_vals) / 2.0
        diff_vals = pred_vals - true_vals  

        mean_diff = np.mean(diff_vals)
        sd_diff = np.std(diff_vals)

        plt.figure(figsize=(5, 4))
        plt.scatter(mean_vals, diff_vals, alpha=0.7)
        plt.axhline(mean_diff, color='red', linestyle='--', label='Mean diff')
        plt.axhline(mean_diff + 1.96*sd_diff, color='gray', linestyle='--', label='±1.96 SD')
        plt.axhline(mean_diff - 1.96*sd_diff, color='gray', linestyle='--')

        plt.xlabel(f'Mean of True and Pred ({m})')
        plt.ylabel(f'Pred - True ({m})')
        plt.title(f'Bland–Altman plot – {m}')
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(iter_dir, f"bland_altman_{m}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Bland–Altman plot saved for {m} → {save_path}")
    
            

    mean_acc = np.mean([r['accuracy'] for r in loso_results])
    std_acc = np.std([r['accuracy'] for r in loso_results])
    mean_kappa = np.mean([r['kappa'] for r in loso_results])
    std_kappa = np.std([r['kappa'] for r in loso_results])

    print("\n" + "="*60)
    print(f"LOSO Cross-Validation Results (10 subjects):")
    print(f"  Accuracy = {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"  Kappa    = {mean_kappa:.3f} ± {std_kappa:.3f}")
    print("="*60)
    print_performance_metrics(np.array(all_y_test), np.array(all_y_pred))
    t = np.arange(len(example_y_true)) 
    stage_names = ['Wake','N1','N2','N3','REM']
    iter_dir = os.path.join("result", f"iteration_{config.CURRENT_ITERATION}")
    os.makedirs(iter_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.step(t, example_y_true, where='post')
    plt.yticks(range(5), stage_names)
    plt.ylabel('Stage')
    plt.title(f'Ground Truth Hypnogram – {example_subject_id}')

    plt.subplot(2, 1, 2)
    plt.step(t, example_y_pred, where='post', color='orange')
    plt.yticks(range(5), stage_names)
    plt.ylabel('Stage')
    plt.title(f'Predicted Hypnogram – {example_subject_id}')
    plt.xlabel('Epochs')

    plt.tight_layout()
    save_path = os.path.join(iter_dir, f"hypnogram_{example_subject_id}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Hypnogram saved to {save_path}")

    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    true_perc = [example_true_metrics[s] for s in stages]
    pred_perc = [example_pred_metrics[s] for s in stages]
    x = np.arange(len(stages))
    width = 0.35
    save_path = os.path.join(iter_dir, f"Stage_percentage_comparison{example_subject_id}.png")
    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, true_perc, width, label='Ground Truth')
    plt.bar(x + width/2, pred_perc, width, label='Predicted')
    plt.xticks(x, stages)
    plt.ylabel('Percentage of TST (%)')
    plt.title(f'Stage percentages – {example_subject_id}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    metrics_to_show = [
    'Sleep_Onset',
    'REM_latency',
    'Total_Sleep_Time',
    'Sleep_Efficiency',
    'WASO',
    'Awakenings'
    ]
    rows = []
    for m in metrics_to_show:
        true_val = example_true_metrics[m]
        pred_val = example_pred_metrics[m]
        error = abs(pred_val - true_val)
        rows.append({
        'Metric': m,
        'Ground Truth': true_val,
        'Predicted': pred_val,
        'Error': error})
    df_metrics = pd.DataFrame(rows)
    print("\nSleep architecture metrics – example subject")
    print(df_metrics.to_string(index=False))
    save_path = os.path.join(iter_dir, f"{example_subject_id}_metrics.csv")
    df_metrics.to_csv(save_path, index=False)
    print(f"Metrics table saved to {save_path}")
    print("\nPer-Subject Performance:")
    for r in sorted(loso_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"  {r['subject']}: {r['accuracy']:.1%} (kappa={r['kappa']:.3f})")
        
        
    final_model_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)), 
    ('scaler', StandardScaler()),
    ('classifier', model) 
    ])

    print("\n" + "="*60)
    print("Training FINAL MODEL on ALL 10 Subjects' Data...")
    final_model_pipeline.fit(features, labels)
    final_scalar = final_model_pipeline.named_steps['scaler']
    print("Final model trained successfully.")
    print("="*60)
    
    scaler_filename = f"scaler_iter{config.CURRENT_ITERATION}.joblib"
    save_cache(final_scalar, scaler_filename, config.CACHE_DIR)
    print(f"Saved final scaler to {config.CACHE_DIR}/{scaler_filename}") 
    
    return final_model_pipeline


def print_performance_metrics(y_true, y_pred):

    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_labels = list(range(5))

    print("\n" + "="*70)
    print("SLEEP STAGE CLASSIFICATION PERFORMANCE METRICS")
    print("="*70)

 
    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Macro F1-Score: {macro_f1:.3f}")
    print(f"Weighted F1-Score: {weighted_f1:.3f}")


    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=stage_labels)
    cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
    print(cm_df.to_string())
    print("\nPer-Class Performance Metrics:")
    print("-" * 70)
    print(f"{'Stage':<8} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1-Score':<10}")
    print("-" * 70)


    for i, stage_name in enumerate(stage_names):
        if i in y_true: 
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((y_pred == i) & (y_true == i)) / np.sum(class_mask)
            else:
                class_accuracy = 0.0
            sensitivity = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]
            tn = np.sum((y_true != i) & (y_pred != i))
            fp = np.sum((y_true != i) & (y_pred == i))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

            print(f"{stage_name:<8} {class_accuracy:<10.3f} {sensitivity:<12.3f} {specificity:<12.3f} {f1:<10.3f}")
        else:
            print(f"{stage_name:<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print("-" * 70)
    print("\nClass Distribution in Test Set:")
    unique, counts = np.unique(y_true, return_counts=True)
    total_samples = len(y_true)

    for stage_idx, count in zip(unique, counts):
        stage_name = stage_names[stage_idx]
        percentage = count / total_samples * 100
        print(f"{stage_name}: {count} samples ({percentage:.1f}%)")

    
def calculate_sleep_metrics(labels, epoch_duration=30):
    print("-----------------------LABELS DEBUG PRINT----------")
    ctr_SOL, ctr_REM, ctr_TST, ctr_TIB, waso_count, awakenings = 0, 0, 0, 0, 0, 0
    stage_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    prev_label = None
    found_sleep_onset, found_REM = False, False
    for l in labels:
        ctr_TIB += 1
     
        stage_counts[l] += 1
        if found_sleep_onset is False:
            if l == 0:
                ctr_SOL += 1
            else:
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
        prev_label = l 
            
    print()
    metrics = {}
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
    WASO = (epoch_duration * waso_count) / 60 
        
    total_epochs = len(labels)
    stage_percentages = {
        'Wake': (stage_counts[0] / total_epochs) * 100,
        'N1': (stage_counts[1] / total_epochs) * 100,
        'N2': (stage_counts[2] / total_epochs) * 100,
        'N3': (stage_counts[3] / total_epochs) * 100,
        'REM': (stage_counts[4] / total_epochs) * 100
    }
    
    metrics = {
        'Sleep_Onset': float(sleep_onset), 
        'REM_latency': float(REM_latency), 
        'Total_Sleep_Time': float(total_sleep_time_h), 
        'Sleep_Efficiency': float(sleep_efficiency), 
        'WASO': float(WASO),  
        'Awakenings': int(awakenings),
        'Wake': float(stage_percentages['Wake']),
        'N1': float(stage_percentages['N1']),
        'N2': float(stage_percentages['N2']),
        'N3': float(stage_percentages['N3']),
        'REM': float(stage_percentages['REM'])
    }

    return metrics
