import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    brier_score_loss
)

def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Computes all standard biomedical classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    brier = brier_score_loss(y_true, y_prob) if y_prob is not None else np.nan
    
    return {
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall / Sensitivity (%)": round(recall * 100, 2),
        "Specificity (%)": round(specificity * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2),
        "ROC-AUC": round(auc, 4) if not np.isnan(auc) else "N/A",
        "Brier Score": round(brier, 4) if not np.isnan(brier) else "N/A",
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

def plot_confusion_matrices(results_dict, output_path):
    """
    Plots side-by-side confusion matrices for key comparison models.
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, res) in zip(axes, results_dict.items()):
        cm = np.array([[res["TN"], res["FP"]], [res["FN"], res["TP"]]])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Healthy (0)", "Disease (1)"],
            yticklabels=["Healthy (0)", "Disease (1)"]
        )
        ax.set_title(f"{name}\nAcc: {res['Accuracy (%)']}%, F1: {res['F1-Score (%)']}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved confusion matrices to {output_path}")

def plot_roc_curves(roc_dict, output_path):
    """
    Plots comparative ROC curves for models.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)")
    
    for name, (y_true, y_prob) in roc_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        lw = 2.5 if "HG-FASNet" in name else 1.5
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=lw)
        
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved ROC curves to {output_path}")

def plot_metrics_comparison_barchart(df_metrics, output_path):
    """
    Plots side-by-side bar chart of Accuracy, Precision, Recall, and F1-Score.
    """
    metrics_to_plot = ["Accuracy (%)", "Precision (%)", "Recall / Sensitivity (%)", "F1-Score (%)"]
    df_plot = df_metrics.set_index("Model")[metrics_to_plot]
    
    ax = df_plot.plot(kind="bar", figsize=(12, 6), width=0.8, colormap="viridis")
    plt.title("Performance Benchmark: Baselines vs. Base Paper vs. Proposed HG-FASNet", fontsize=14, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=12)
    plt.ylim(65, 100)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    
    # Annotate top model values
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved metrics comparison barchart to {output_path}")
