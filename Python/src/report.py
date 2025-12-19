import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    cohen_kappa_score,
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    matthews_corrcoef
)
import seaborn as sns
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
output_dir = "result/reports"   
os.makedirs(output_dir, exist_ok=True)
def generate_report(model, selected_features, combined_labels, config, txt_filename=None):

    iteration = getattr(config, "CURRENT_ITERATION", "unknown")

    if txt_filename is None:
        txt_filename = os.path.join(
        output_dir, f"report_iteration_{iteration}.txt"
    )

    y_pred = model.predict(selected_features)
    y_true = combined_labels
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)


    if hasattr(config, "SLEEP_CLASSES"):
        classes = config.SLEEP_CLASSES
    else:
        classes = np.unique(y_true)

    report_lines = []
    report_lines.append("===== OVERALL METRICS =====")
    report_lines.append(f"Accuracy: {acc:.4f}")
    report_lines.append(f"Cohen's Kappa: {kappa:.4f}")
    report_lines.append(f"Macro F1: {f1_macro:.4f}")
    report_lines.append(f"Micro F1: {f1_micro:.4f}")
    report_lines.append(f"Weighted F1: {f1_weighted:.4f}")
    report_lines.append(f"Matthews Correlation Coefficient: {mcc:.4f}")
    report_lines.append("\n===== PER-CLASS METRICS =====")
    for i, cls in enumerate(classes):
        report_lines.append(f"{cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_per_class[i]:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    report_lines.append("\n===== CONFUSION MATRIX =====")
    report_lines.append(str(cm))

    with open(txt_filename, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to {txt_filename}")
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
