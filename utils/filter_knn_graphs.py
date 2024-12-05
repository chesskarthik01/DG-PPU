import torch
import numpy as np
import h5py


def filter_point_cloud(pos, labels, knn_graphs, k=20, verbose=False):
    """
    Filter points based on k-NN neighbors and their labels, and return filtered points and their indices.
    Args:
        pos (torch.Tensor): Point cloud coordinates (num_points, 3)
        labels (torch.Tensor): Labels for each point (num_points,)
        knn_graphs (torch.Tensor): The k-NN graph (edge_index) from the model (2, num_edges)
        k (int): The number of nearest neighbors to check (default is 20)
        verbose (bool): Whether to print detailed information during filtering

    Returns:
        filtered_pos (torch.Tensor): Filtered point cloud coordinates (num_filtered_points, 3)
        filtered_labels (torch.Tensor): Filtered labels (num_filtered_points,)
        keep_indices (torch.Tensor): Indices of the points that were kept after filtering.
    """
    num_points = pos.size(0)
    keep_mask = torch.zeros(num_points, dtype=torch.bool)

    for i in range(num_points):
        # Get the indices of the neighbors of point i from the k-NN graph (both directions)
        neighbors = torch.cat([
            knn_graphs[1][knn_graphs[0] == i],  # neighbors where i is source
            knn_graphs[0][knn_graphs[1] == i]   # neighbors where i is target
        ]).unique()

        # Check if the point has at least k neighbors
        if len(neighbors) < k:
            continue

        # Among the k nearest neighbors, check if they all have the same label
        nearest_neighbors = neighbors[:k]  # Take the first k neighbors
        same_label_neighbors = (
            labels[nearest_neighbors] == labels[i]).sum().item()

        # Keep the point only if all k neighbors share the same label
        if same_label_neighbors == k:
            keep_mask[i] = True

    # Filter the point cloud coordinates and labels based on the keep_mask
    filtered_pos = pos[keep_mask]
    filtered_labels = labels[keep_mask]

    # Return indices of the kept points
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze()

    if verbose:
        print(f"Filtered {filtered_pos.size(0)} points out of {num_points}")

    return filtered_pos, filtered_labels, keep_indices


def save_filtered_point_cloud_h5(filtered_pos, filtered_labels, output_path, num_points=1024):
    """
    Save the filtered point clouds and labels to an HDF5 file in a format compatible with DCP's DGCNN.

    Args:
        filtered_pos (torch.Tensor): Filtered point cloud coordinates (num_filtered_points, 3).
        filtered_labels (torch.Tensor): Filtered labels (num_filtered_points,).
        output_path (str): Path to save the filtered point clouds (.h5 file).
        num_points (int): Number of points per point cloud (default: 1024).
    """
    # Ensure the data is on CPU
    filtered_pos_np = filtered_pos.cpu().numpy()
    filtered_labels_np = filtered_labels.cpu().numpy()

    # Create batches of fixed size (num_points) for compatibility with the DCP DGCNN model
    num_samples = len(filtered_pos_np) // num_points
    reshaped_pos = filtered_pos_np[:num_samples *
                                   num_points].reshape(num_samples, num_points, 3)
    reshaped_labels = filtered_labels_np[:num_samples *
                                         num_points].reshape(num_samples, num_points)

    # Save to HDF5 file in the correct format
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data', data=reshaped_pos.astype(
            'float32'))  # Point clouds
        f.create_dataset('label', data=reshaped_labels[:, 0].astype(
            'int64'))  # Labels (1 per cloud)

    print(f"Filtered point clouds saved to {output_path} in HDF5 format.")
