import torch
import numpy as np


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


def save_filtered_point_cloud(filtered_pos, filtered_labels, output_path):
    """
    Save the filtered point cloud and labels to an .npz file.

    Args:
        filtered_pos (torch.Tensor): Filtered point cloud coordinates.
        filtered_labels (torch.Tensor): Filtered point cloud labels.
        output_path (str): Path to save the filtered point cloud (.npz file).
    """
    # Convert tensors to numpy arrays
    filtered_pos_np = filtered_pos.cpu().numpy()
    filtered_labels_np = filtered_labels.cpu().numpy()

    # Save data as an .npz file
    np.savez_compressed(output_path, positions=filtered_pos_np,
                        labels=filtered_labels_np)
    print(f"Filtered point cloud saved to {output_path}.npz")
