"""Pedagogical GNN-LRP implementation for dense GCN graph classifiers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dense_gcn import DenseGCNGraphClassifier


@dataclass
class GNNLRPResult:
    class_index: int
    class_probability: float
    node_relevance: np.ndarray
    edge_relevance: np.ndarray
    feature_relevance: np.ndarray


def _normalize_nonnegative(vector: torch.Tensor, epsilon: float) -> torch.Tensor:
    vector = torch.clamp(vector, min=0.0)
    total = vector.sum()
    if float(total.item()) <= epsilon:
        return torch.full_like(vector, 1.0 / vector.numel())
    return vector / total


def _propagate_mean_pool_relevance(
    node_embeddings: torch.Tensor,
    pooled_relevance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    positive_embeddings = torch.clamp(node_embeddings, min=0.0)
    denom = positive_embeddings.sum(dim=0, keepdim=True).clamp(min=epsilon)
    return (positive_embeddings / denom) * pooled_relevance.unsqueeze(0)


def _propagate_gcn_layer_relevance(
    input_features: torch.Tensor,
    normalized_adjacency: torch.Tensor,
    weight: torch.Tensor,
    output_relevance: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes, in_features = input_features.shape
    out_features = output_relevance.shape[1]
    positive_inputs = torch.clamp(input_features, min=0.0)
    positive_weights = torch.clamp(weight.t(), min=0.0)
    input_relevance = torch.zeros_like(positive_inputs)
    edge_relevance = torch.zeros(
        (num_nodes, num_nodes),
        dtype=positive_inputs.dtype,
        device=positive_inputs.device,
    )

    for output_node in range(num_nodes):
        for output_feature in range(out_features):
            relevance_value = output_relevance[output_node, output_feature]
            if float(relevance_value.item()) <= 0.0:
                continue
            contributions = (
                normalized_adjacency[output_node][:, None]
                * positive_inputs
                * positive_weights[:, output_feature][None, :]
            )
            normalization = contributions.sum()
            if float(normalization.item()) <= epsilon:
                fallback = normalized_adjacency[output_node][:, None].expand(num_nodes, in_features)
                fallback = torch.clamp(fallback, min=0.0)
                fallback_normalization = fallback.sum().clamp(min=epsilon)
                redistributed = fallback / fallback_normalization * relevance_value
            else:
                redistributed = contributions / normalization * relevance_value
            input_relevance += redistributed
            edge_relevance[:, output_node] += redistributed.sum(dim=1)

    return input_relevance, edge_relevance


def explain_graph_with_gnn_lrp(
    model: DenseGCNGraphClassifier,
    adjacency: np.ndarray,
    node_features: np.ndarray,
    target_class: Optional[int] = None,
    epsilon: float = 1e-9,
) -> GNNLRPResult:
    model.eval()
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    features_tensor = torch.tensor(node_features, dtype=torch.float32)
    with torch.no_grad():
        cache = model.forward_with_cache(adjacency_tensor, features_tensor)

        probabilities = torch.softmax(cache.logits, dim=0)
        class_index = int(cache.logits.argmax().item()) if target_class is None else int(target_class)
        class_probability = float(probabilities[class_index].item())

        classifier_weight = torch.clamp(model.classifier.weight[class_index], min=0.0)
        graph_basis = _normalize_nonnegative(cache.graph_embedding * classifier_weight, epsilon)
        pooled_relevance = graph_basis * class_probability

        output_relevance = _propagate_mean_pool_relevance(
            node_embeddings=cache.layer_outputs[-1],
            pooled_relevance=pooled_relevance,
            epsilon=epsilon,
        )

        total_edge_relevance = torch.zeros_like(adjacency_tensor)
        for layer_index in range(len(model.layers) - 1, -1, -1):
            layer = model.layers[layer_index]
            input_relevance, edge_relevance = _propagate_gcn_layer_relevance(
                input_features=cache.layer_inputs[layer_index],
                normalized_adjacency=cache.normalized_adjacency,
                weight=layer.linear.weight.detach(),
                output_relevance=output_relevance,
                epsilon=epsilon,
            )
            total_edge_relevance += edge_relevance
            output_relevance = input_relevance

        node_relevance = output_relevance.sum(dim=1)
        feature_relevance = output_relevance.sum(dim=0)
        total_edge_relevance = 0.5 * (total_edge_relevance + total_edge_relevance.t())
        total_edge_relevance = total_edge_relevance * adjacency_tensor
        total_edge_relevance.fill_diagonal_(0.0)

    return GNNLRPResult(
        class_index=class_index,
        class_probability=class_probability,
        node_relevance=node_relevance.detach().cpu().numpy(),
        edge_relevance=total_edge_relevance.detach().cpu().numpy(),
        feature_relevance=feature_relevance.detach().cpu().numpy(),
    )
