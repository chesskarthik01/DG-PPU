import torch
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, knn_graph
from torch_geometric.data import Data


class DGCNNWithKNN(torch.nn.Module):
    def __init__(self, k=20, num_classes=3):
        super(DGCNNWithKNN, self).__init__()
        self.k = k

        # Define EdgeConv layers
        self.conv1 = EdgeConv(self.mlp(6, 64), aggr='max')
        self.conv2 = EdgeConv(self.mlp(128, 64), aggr='max')
        self.conv3 = EdgeConv(self.mlp(128, 64), aggr='max')
        self.conv4 = EdgeConv(self.mlp(128, 128), aggr='max')

        # MLP layers for classification
        self.mlp1 = torch.nn.Sequential(
            # Concatenated feature size: 320 (64+64+64+128)
            torch.nn.Linear(320, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.mlp3 = torch.nn.Linear(128, num_classes)

    def mlp(self, input_dim, output_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, output_dim)
        )

    def forward(self, data):
        pos, batch = data.pos, data.batch

        # Store k-NN graphs from each layer
        knn_graphs = []

        # First EdgeConv layer with k-NN graph
        edge_index1 = knn_graph(pos, self.k, batch=batch)
        knn_graphs.append(edge_index1)
        x1 = self.conv1(pos, edge_index1)

        # Second EdgeConv layer with k-NN graph
        edge_index2 = knn_graph(x1, self.k, batch=batch)
        knn_graphs.append(edge_index2)
        x2 = self.conv2(x1, edge_index2)

        # Third EdgeConv layer with k-NN graph
        edge_index3 = knn_graph(x2, self.k, batch=batch)
        knn_graphs.append(edge_index3)
        x3 = self.conv3(x2, edge_index3)

        # Fourth EdgeConv layer with k-NN graph
        edge_index4 = knn_graph(x3, self.k, batch=batch)
        knn_graphs.append(edge_index4)
        x4 = self.conv4(x3, edge_index4)

        # Concatenate learned features from different layers
        # Total feature size: 320 (64+64+64+128)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        # Final MLP for classification
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)  # Output logits for each point

        return x, knn_graphs  # Return the logits and k-NN graphs
