import torch
import os
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from models.dgcnn import DGCNNWithKNN
from utils.dataset import CATMAUS
from utils.filter_knn_graphs import filter_point_cloud, save_filtered_point_cloud_h5
from utils.plot_knn_graph_comparison import plot_knn_graphs_side_by_side
from utils.plot_confusion_matrices import log_confusion_matrices
from utils.plot_graph_edges import plot_knn_2d_graph
from torch_geometric.data import Data
import yaml

# Set up device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load configuration from 'config.yaml'
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load the pre-trained model
model_path = os.path.join('outputs', 'trained_models', 'trained_model.pth')
model = DGCNNWithKNN(k=20, num_classes=3).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode


def load_data_and_filter_point_cloud(model, data_loader, verbose=True):
    """
    Passes data through the model to extract and filter point clouds.
    Args:
        model: Trained DGCNNWithKNN model.
        data_loader: DataLoader to pass batches through the model.
        verbose (bool): Whether to print detailed information during processing.

    Returns:
        unfiltered_positions, unfiltered_labels: Unfiltered positions and labels.
        filtered_positions, filtered_labels: Filtered positions and labels.
        unfiltered_pred_labels: Unfiltered predictions from the model.
        ground_truth_labels: Ground truth labels from the dataset.
        filtered_ground_truth_labels: Ground truth labels corresponding to the filtered points.
    """
    all_unfiltered_pos = []  # Unfiltered point cloud positions
    all_unfiltered_pred_labels = []  # Unfiltered label predictions
    all_filtered_pos = []  # Filtered point cloud positions
    all_filtered_labels = []  # Filtered point cloud labels
    all_filtered_ground_truths = []  # Ground truth labels for filtered points
    all_ground_truths_pos = []  # Grount truth positions for reference
    all_ground_truths = []  # Ground truth labels for reference

    total_batches = len(data_loader)
    print(f"Starting to process {total_batches} batches of data...")

    with torch.no_grad():  # No need to calculate gradients
        for batch_idx, data in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{total_batches}...")

            data = data.to(device)

            # Forward pass to get model predictions and k-NN graph
            unfiltered_preds, knn_graphs = model(data)

            # Save unfiltered data
            all_unfiltered_pos.append(data.pos.cpu().numpy())
            all_unfiltered_pred_labels.append(unfiltered_preds.argmax(
                dim=1).cpu().numpy())  # Predicted labels before filtering
            all_ground_truths.append(
                data.y.cpu().numpy())  # Ground truth labels

            # Save ground truth positions
            all_ground_truths_pos.append(data.pos.cpu().numpy())

            # Filter the point cloud based on the k-NN graph and labels
            filtered_pos, filtered_labels, keep_indices = filter_point_cloud(
                data.pos, data.y, knn_graphs, verbose=True)

            # Accumulate filtered positions and labels
            all_filtered_pos.append(filtered_pos.cpu().numpy())
            all_filtered_labels.append(filtered_labels.cpu().numpy())

            # Get ground truth labels corresponding to the filtered points
            filtered_ground_truths = data.y[keep_indices].cpu().numpy()
            all_filtered_ground_truths.append(filtered_ground_truths)

            if verbose:
                print(
                    f"Batch {batch_idx + 1} processed. Filtered {len(filtered_pos)} points.")

    # Combine all batches into a single point cloud
    unfiltered_pos = np.concatenate(all_unfiltered_pos, axis=0)
    unfiltered_pred_labels = np.concatenate(all_unfiltered_pred_labels, axis=0)
    filtered_pos = np.concatenate(all_filtered_pos, axis=0)
    filtered_labels = np.concatenate(all_filtered_labels, axis=0)
    ground_truth_pos = np.concatenate(all_ground_truths_pos, axis=0)
    ground_truth_labels = np.concatenate(all_ground_truths, axis=0)
    filtered_ground_truth_labels = np.concatenate(
        all_filtered_ground_truths, axis=0)

    return unfiltered_pos, unfiltered_pred_labels, filtered_pos, filtered_labels, ground_truth_pos, ground_truth_labels, filtered_ground_truth_labels


def extract_feature_vectors(model, data_loader, verbose=True):
    """
    Extracts the feature vectors from the model using a single forward pass
    over the entire dataset combined from all batches in the data loader.

    Args:
        model: Trained DGCNNWithKNN model.
        data_loader: DataLoader containing the test data.
        verbose (bool): Whether to print detailed information during processing.

    Returns:
        final_feature_vectors (torch.Tensor): Extracted feature vectors for all points in the dataset.
    """

    all_positions = []
    all_features = []
    all_batches = []

    print(f'Starting to process {len(data_loader)} batches of data...')

    # Iterate over the DataLoader to collect all batches
    for batch_idx, batch_data in enumerate(data_loader):
        batch_data = batch_data.to(device)  # Use device from main()

        # Check if the features (X) are defined
        if batch_data.x is not None:
            all_features.append(batch_data.x)
        else:
            print(
                f"Warning: Batch {batch_idx} has no 'X' attribute. Using default zero tensor.")
            # Use a default tensor if features are missing (adjust dimensions as needed)
            all_features.append(torch.zeros(
                (batch_data.num_nodes, 3)).to(device))

        # Append positions and batches normally
        all_positions.append(batch_data.pos)
        all_batches.append(batch_data.batch + batch_idx *
                           batch_data.batch.max() + 1)

    # Concatenate all data to form a single Data object
    positions = torch.cat(all_positions, dim=0).to(device)
    features = torch.cat(all_features, dim=0).to(device)
    batch = torch.cat(all_batches, dim=0).to(device)

    # Print the total number of points
    if verbose:
        print(
            f"Total number of points across all batches: {positions.shape[0]}")

    # Create a combined Data object for all points
    data = Data(x=features, pos=positions, batch=batch)

    # Perform a forward pass to get the final feature vectors for all points
    with torch.no_grad():
        model_output = model(data)

        # Assuming model_output is the final feature vectors
        if isinstance(model_output, tuple):
            final_feature_vectors = model_output[0]
        else:
            final_feature_vectors = model_output

    # Return the extracted feature vectors
    return final_feature_vectors.cpu().numpy()


def main(verbose=True):
    # Dataset root path
    root = config['root']

    # Create dataset and data loader for testing
    dataset = CATMAUS(root=root, monte_carlo=False, batch_size=1024)
    print("Dataset loaded successfully")

    # Create test data loader
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"Test data loader created: {len(test_loader)} batches")

    # Extract feature vectors for the entire test dataset
    feature_vectors = extract_feature_vectors(model, test_loader, verbose=True)

    # Load data, filter, and get unfiltered and filtered point clouds
    unfiltered_pos, unfiltered_labels, filtered_pos, filtered_labels, ground_truth_pos, ground_truth_labels, filtered_ground_truth_labels = load_data_and_filter_point_cloud(
        model, test_loader, verbose=verbose)

    # Save the filtered point clouds to a .npz file for future use
    output_path = 'outputs/point_clouds/test_filtered_point_clouds.h5'
    save_filtered_point_cloud_h5(torch.tensor(
        filtered_pos), torch.tensor(filtered_labels), output_path)

    # Plot the ground truth point clouds
    plot_knn_graphs_side_by_side(ground_truth_pos, ground_truth_labels, filtered_pos, filtered_labels,
                                 save_path='outputs/graphs/test_3D_point_cloud_comparison'
                                 )

    # Compute and log confusion matrices using Seaborn
    log_confusion_matrices(filtered_labels, unfiltered_labels, ground_truth_labels, filtered_ground_truth_labels, num_classes=3,
                           save_path='outputs/confusion_matrices/test_confusion_matrix_comparison')

    # Plot filtered k-NN graphs with edges
    plot_knn_2d_graph(feature_vectors, k=20,
                      save_path='outputs/graphs/test_kNN_graph_with_edges')
    print("Processing complete and data saved successfully.")


if __name__ == "__main__":
    main(verbose=True)  # Set verbose=True to print output at the end of each batch
