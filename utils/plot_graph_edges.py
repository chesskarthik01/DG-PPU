import os
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import knn_graph
import torch


def plot_knn_2d_graph(feature_vectors, k=20, title="Filtered k-NN graph with edges", save_path=None):
    """
    Plot the k-NN graph using the first two dimensions of the feature vectors in 2D.
    """
    # Checking and converting feature_vectors into torch tensors
    if not isinstance(feature_vectors, torch.Tensor):
        feature_vectors = torch.tensor(feature_vectors)

    # Compute k-NN graph
    edge_index = knn_graph(feature_vectors, k=k).t().numpy()

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes with positions
    for i, pos in enumerate(feature_vectors.cpu().numpy()):
        # Using the first 2 components for 2D projection
        G.add_node(i, pos=(pos[0], pos[1]))

    # Add edges based on kNN connections
    G.add_edges_from(edge_index)

    # Plot the graph
    plt.figure(figsize=(15, 12))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_size=50, node_color='skyblue', width=5,
            edge_color='gray', with_labels=False)
    plt.title(title)

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.png")
        print(f"Static plot saved at {save_path}.png")

    plt.close()
