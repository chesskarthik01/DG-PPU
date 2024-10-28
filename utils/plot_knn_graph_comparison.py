import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import KDTree
import plotly.colors as pc


def plot_knn_graphs_side_by_side(ground_truth_pos, ground_truth_labels, filtered_pos, filtered_labels, title="k-NN Graph Comparison", save_path=None):
    """
    Plots the ground_truth and filtered point clouds side by side using Plotly and saves them as an interactive HTML file.

    Args:
        ground_truth_pos (numpy.ndarray): Ground truth point coordinates.
        ground_truth_labels (numpy.ndarray): Ground truth point labels.
        filtered_pos (numpy.ndarray): Filtered point coordinates.
        filtered_labels (numpy.ndarray): Filtered point labels.
        title (str): Title of the entire subplot figure.
        save_path (str): Path to save the interactive plot as an .html file.
    """
    # Color scale
    color_scale = pc.qualitative.G10

    # Create a subplot with 1 row and 2 columns for side-by-side comparison, both as 3D subplots
    fig = make_subplots(
        rows=1, cols=2,
        # Specify 3D plots
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("Unfiltered Point Cloud", "Filtered Point Cloud")
    )

    # Helper function to add edges based on k-NN graph
    def add_edges(positions, k):
        kdtree = KDTree(positions)
        edges = []
        for i, pos in enumerate(positions):
            # +1 because the first neighbor is the point itself
            distances, neighbors = kdtree.query(pos, k=k + 1)
            for neighbor in neighbors[1:]:  # Skip the first neighbor (itself)
                edges.append((i, neighbor))

        edge_trace = []
        for edge in edges:
            x0, y0, z0 = positions[edge[0]]
            x1, y1, z1 = positions[edge[1]]
            edge_trace.append(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode='lines',
                line=dict(color='gray', width=1),
                hoverinfo='none',
                showlegend=False  # Hide from legend
            ))

        return edge_trace

    # Unfiltered point cloud scatter plot
    ground_truth_scatter = go.Scatter3d(
        x=ground_truth_pos[:, 0],
        y=ground_truth_pos[:, 1],
        z=ground_truth_pos[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=ground_truth_labels,
            colorscale=color_scale,
            opacity=0.8
        ),
        name="Unfiltered"
    )

    # Filtered point cloud scatter plot
    filtered_scatter = go.Scatter3d(
        x=filtered_pos[:, 0],
        y=filtered_pos[:, 1],
        z=filtered_pos[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=filtered_labels,
            colorscale=color_scale,
            opacity=0.8
        ),
        name="Filtered"
    )

    # Add the ground_truth and filtered scatter plots to the respective subplots
    fig.add_trace(ground_truth_scatter, row=1, col=1)
    fig.add_trace(filtered_scatter, row=1, col=2)

    # Set the overall title for the plot
    fig.update_layout(title_text=title)

    # Save the figure as an HTML file
    if save_path:
        fig.write_html(f"{save_path}.html")
        print(f"Interactive plot saved as {save_path}.html")

    return fig  # Return the figure object for display if needed
