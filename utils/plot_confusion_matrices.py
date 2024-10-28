import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os


def log_confusion_matrices(filtered_preds, unfiltered_preds, ground_truth_labels, filtered_ground_truth_labels, num_classes, save_path=None):
    """
    Computes and logs confusion matrices for ground truth, filtered, and unfiltered predictions using Seaborn.

    Args:
        filtered_preds (numpy.ndarray): Predictions after filtering.
        unfiltered_preds (numpy.ndarray): Predictions before filtering.
        ground_truth_labels (numpy.ndarray): Ground truth labels for unfiltered data.
        filtered_ground_truth_labels (numpy.ndarray): Ground truth labels for filtered data.
        num_classes (int): Number of classes in the dataset.
        save_path (str): Path to save the confusion matrix plot as an image.

    Returns:
        None
    """
    # Compute confusion matrices
    cm_ground_truth = confusion_matrix(
        ground_truth_labels, ground_truth_labels, labels=range(num_classes))  # Diagonal matrix for perfect match
    cm_unfiltered = confusion_matrix(
        ground_truth_labels, unfiltered_preds, labels=range(num_classes))
    cm_filtered = confusion_matrix(
        filtered_ground_truth_labels, filtered_preds, labels=range(num_classes))

    # Create a figure with three subplots for ground truth, unfiltered, and filtered confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Ground truth confusion matrix (this will just show a perfect diagonal matrix)
    sns.heatmap(cm_ground_truth, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Ground Truth Confusion Matrix\n",
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel("\nPredicted", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Actual", fontsize=12, fontweight='bold')

    # Unfiltered confusion matrix
    sns.heatmap(cm_unfiltered, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title("Unfiltered Predictions Confusion Matrix\n",
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel("\nPredicted", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Actual", fontsize=12, fontweight='bold')

    # Filtered confusion matrix
    sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title("Filtered Predictions Confusion Matrix\n",
                      fontsize=12, fontweight='bold')
    axes[2].set_xlabel("\nPredicted", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Actual\n", fontsize=12, fontweight='bold')

    # Add more horizontal space between the plots
    plt.subplots_adjust(wspace=0.4)

    # Adjust the space between the plots
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.png")
        print(f"Confusion matrices saved at {save_path}.png")

    plt.close()
