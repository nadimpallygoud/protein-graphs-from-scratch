"""Dense GCN model used for protein graph classification and GNN-LRP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor, nn


def normalize_adjacency_torch(adjacency: Tensor, add_self_loops: bool = True) -> Tensor:
    if add_self_loops:
        adjacency = adjacency + torch.eye(adjacency.size(0), device=adjacency.device)
    degree = adjacency.sum(dim=1)
    inv_sqrt_degree = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
    return inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]


class DenseGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, normalized_adjacency: Tensor, features: Tensor) -> Tensor:
        return normalized_adjacency @ self.linear(features)


@dataclass
class ForwardCache:
    normalized_adjacency: Tensor
    layer_inputs: List[Tensor]
    layer_pre_activations: List[Tensor]
    layer_outputs: List[Tensor]
    graph_embedding: Tensor
    logits: Tensor


class DenseGCNGraphClassifier(nn.Module):
    """Dense graph classifier kept explicit so explanation rules remain inspectable."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("DenseGCNGraphClassifier expects at least two layers")
        layers: List[DenseGCNLayer] = [DenseGCNLayer(in_features, hidden_features)]
        for _ in range(num_layers - 2):
            layers.append(DenseGCNLayer(hidden_features, hidden_features))
        layers.append(DenseGCNLayer(hidden_features, hidden_features))
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_features, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def encode(self, adjacency: Tensor, features: Tensor) -> Tensor:
        normalized_adjacency = normalize_adjacency_torch(adjacency)
        hidden = features
        for layer in self.layers:
            hidden = self.activation(layer(normalized_adjacency, hidden))
            hidden = self.dropout(hidden)
        return hidden

    def forward(self, adjacency: Tensor, features: Tensor) -> Tensor:
        node_embeddings = self.encode(adjacency, features)
        graph_embedding = node_embeddings.mean(dim=0)
        return self.classifier(graph_embedding)

    def forward_with_cache(self, adjacency: Tensor, features: Tensor) -> ForwardCache:
        normalized_adjacency = normalize_adjacency_torch(adjacency)
        layer_inputs: List[Tensor] = []
        layer_pre_activations: List[Tensor] = []
        layer_outputs: List[Tensor] = []

        hidden = features
        for layer in self.layers:
            layer_inputs.append(hidden)
            pre_activation = layer(normalized_adjacency, hidden)
            layer_pre_activations.append(pre_activation)
            hidden = self.activation(pre_activation)
            layer_outputs.append(hidden)
        graph_embedding = hidden.mean(dim=0)
        logits = self.classifier(graph_embedding)
        return ForwardCache(
            normalized_adjacency=normalized_adjacency,
            layer_inputs=layer_inputs,
            layer_pre_activations=layer_pre_activations,
            layer_outputs=layer_outputs,
            graph_embedding=graph_embedding,
            logits=logits,
        )


def predict_probability(logits: Tensor, class_index: int) -> float:
    probabilities = torch.softmax(logits, dim=0)
    return float(probabilities[class_index].item())

