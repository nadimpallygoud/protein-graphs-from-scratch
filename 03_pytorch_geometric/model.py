"""PyTorch Geometric GCN baseline."""

from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.nn import GCNConv


class PyGNodeClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.activation(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

