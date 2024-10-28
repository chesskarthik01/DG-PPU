import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from models.dgcnn import DGCNNWithKNN
from utils.dataset import CATMAUS
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Set up device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load configuration from 'config.yaml'
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Add your API key here
wandb.login(key=config['wandb']['api_key'])

# Initialize WandB
wandb.init(
    project=config['wandb']['project_name'],  # Your project name
    name=config['wandb']['run_name'],  # Name for this specific run
    config={
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 100
    }
)


class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(
                        f"Early stopping triggered after {self.counter} epochs of no improvement.")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves the mode's state dict in memory when validation loss decreases
        """
        if self.verbose:
            print(
                f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model state...)')
        self.best_model_state = model.state_dict()


# Function to compute metrics


def compute_metrics(pred, target, num_classes):
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    iou_per_class = []
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for class_id in range(num_classes):
        intersection = ((pred == class_id) & (target == class_id)).sum().item()
        union = ((pred == class_id) | (target == class_id)).sum().item()

        precision = precision_score(
            target_np == class_id, pred_np == class_id, zero_division=0)
        recall = recall_score(target_np == class_id,
                              pred_np == class_id, zero_division=0)
        f1 = f1_score(target_np == class_id, pred_np ==
                      class_id, zero_division=0)
        iou = intersection / union if union != 0 else float('nan')

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        iou_per_class.append(iou)

    return iou_per_class, precision_per_class, recall_per_class, f1_per_class

# Function to log segmentation examples in one combined image


def log_combined_segmentation_to_wandb(all_data_pos, all_preds, all_targets, epoch):
    """
    Logs segmentation examples to WandB in 3D using wandb.Object3D for visualizing point clouds.
    Combines point cloud, ground truth, and predictions into a WandB table.
    """

    # Combine the position data and labels for logging
    pos_with_labels_gt = np.hstack(
        (all_data_pos, all_targets.reshape(-1, 1)))  # Ground truth labels
    pos_with_labels_pred = np.hstack(
        (all_data_pos, all_preds.reshape(-1, 1)))  # Predicted labels

    # Create a WandB table to log both ground truth and predictions alongside the point cloud
    table = wandb.Table(columns=["Point Cloud", "Ground Truth", "Prediction"])

    # Add rows to the table with 3D visualizations
    table.add_data(wandb.Object3D(all_data_pos),  # Original point cloud
                   wandb.Object3D(pos_with_labels_gt),  # Ground truth labels
                   wandb.Object3D(pos_with_labels_pred))  # Predicted labels

    # Log the table to WandB
    wandb.log({f"Segmentation Examples - Epoch {epoch}": table})

# Log confusion matrix to WandB


def log_confusion_matrix_to_wandb(all_preds, all_targets, num_classes, mean_iou, mean_precision, mean_recall, mean_f1, epoch):
    cm = confusion_matrix(all_targets, all_preds)

    # Create a figure with two subplots: one for the confusion matrix, one for the stats
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={
                           'height_ratios': [4, 1]})

    # Plot the confusion matrix in the first subplot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(num_classes), yticklabels=range(num_classes), ax=ax[0])
    ax[0].set_title("Confusion Matrix\n", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("\nPredicted Labels", fontsize=12, fontweight='bold')
    ax[0].set_ylabel("True Labels\n", fontsize=12, fontweight='bold')

    # Create the summary text for the metrics (IoU, Precision, Recall, F1-score)
    stats_text = (
        f"IoU = {mean_iou:.4f}    "
        f"Precision = {mean_precision:.4f}    "
        f"Recall = {mean_recall:.4f}    "
        f"F1 Score = {mean_f1:.4f}"
    )

    # Plot the summary text in the second subplot
    ax[1].axis('off')  # Turn off the axis for the text
    ax[1].text(0.5, 0.5, stats_text, ha="center",
               va="center", fontsize=12, fontweight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Log confusion matrix with the epoch number
    wandb.log({f"Confusion Matrix - Epoch {epoch}": wandb.Image(plt)})
    plt.close()


# Train function


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out, knn_graphs = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Optionally log k-NN graphs here if needed
        print(f'Epoch {epoch} | k-NN Graphs from batch: {knn_graphs}')

    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss

# Test function


def test(model, test_loader, num_classes, criterion, epoch):
    model.eval()
    correct = 0
    total_points = 0
    total_loss = 0

    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    all_preds = []
    all_targets = []
    all_data_pos = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            out, knn_graphs = model(data)
            loss = criterion(out, data.y)
            pred = out.argmax(dim=1)

            total_loss += loss.item()
            correct += (pred == data.y).sum().item()
            total_points += data.y.size(0)

            iou_per_class, precision_per_class, recall_per_class, f1_per_class = compute_metrics(
                pred, data.y, num_classes)
            iou_list.append(iou_per_class)
            precision_list.append(precision_per_class)
            recall_list.append(recall_per_class)
            f1_list.append(f1_per_class)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
            all_data_pos.append(data.pos.cpu().numpy())

            # Optionally log k-NN graphs here if needed
            print(f'k-NN Graphs from batch: {knn_graphs}')

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total_points

    mean_iou = np.nanmean(np.array(iou_list), axis=0)
    mean_precision = np.nanmean(np.array(precision_list), axis=0)
    mean_recall = np.nanmean(np.array(recall_list), axis=0)
    mean_f1 = np.nanmean(np.array(f1_list), axis=0)

    # Flatten predictions and targets for the entire test set for this epoch
    all_preds_flat = np.concatenate(all_preds)
    all_targets_flat = np.concatenate(all_targets)
    all_data_pos_flat = np.concatenate(all_data_pos)

    # Log segmentation examples for this epoch
    log_combined_segmentation_to_wandb(
        all_data_pos_flat, all_preds_flat, all_targets_flat, epoch=epoch)

    # Log confusion matrix for this epoch
    log_confusion_matrix_to_wandb(
        all_preds_flat,
        all_targets_flat,
        num_classes,
        mean_iou=np.mean(mean_iou),
        mean_precision=np.mean(mean_precision),
        mean_recall=np.mean(mean_recall),
        mean_f1=np.mean(mean_f1),
        epoch=epoch
    )

    return {
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        'mean_iou': mean_iou.tolist(),
        'mean_precision': mean_precision.tolist(),
        'mean_recall': mean_recall.tolist(),
        'mean_f1': mean_f1.tolist()
    }

# Main function


def main():
    # Define root directory that contains the dataset files
    root = config['root']

    # Create dataset
    dataset = CATMAUS(root=root, num_points=1024, iterations=500)
    print("Dataset loaded successfully")

    # Split dataset into training and testing
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(
        f"Data loaders created: {len(train_loader)} train batches, {len(test_loader)} test batches")

    # Initialize the model
    model = DGCNNWithKNN(k=20, num_classes=3).to(device)
    print("Model initialized successfully")

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize early stopping
    early_stopping = EarlyStopping(verbose=True)

    num_epochs = wandb.config.num_epochs
    for epoch in range(1, num_epochs + 1):
        print(f"Starting epoch {epoch}")

        # Train and log train loss
        train_loss = train(model, train_loader, optimizer, criterion, epoch)

        # Test and log validation metrics for this epoch
        metrics = test(model, test_loader, num_classes=3,
                       criterion=criterion, epoch=epoch)

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')
        print(
            f'Test Loss: {metrics["test_loss"]:.4f}, Accuracy: {metrics["test_accuracy"]:.4f}')
        print(f'Mean IoU: {metrics["mean_iou"]}')
        print(f'Mean Precision: {metrics["mean_precision"]}')
        print(f'Mean Recall: {metrics["mean_recall"]}')
        print(f'Mean F1-Score: {metrics["mean_f1"]}')

        # Log validation loss and accuracy in one chart
        wandb.log({
            "epoch": epoch,
            "validation_loss": metrics["test_loss"],
            "validation_accuracy": metrics["test_accuracy"]
        })

        # Early stopping based on validation loss
        early_stopping(metrics['test_loss'], model)
        if early_stopping.early_stop:
            break

    # Save the final model
    torch.save(early_stopping.best_model_state,
               "outputs/trained_models/trained_model.pth")

    print("Final model saved after completing all epochs.")

    wandb.finish()


if __name__ == "__main__":
    main()
