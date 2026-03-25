"""Synthetic graphs used in the early educational modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SyntheticNodeClassificationData:
    adjacency: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


def generate_sbm_graph(
    num_nodes_per_class: int = 40,
    p_in: float = 0.25,
    p_out: float = 0.03,
    feature_noise: float = 0.15,
    seed: int = 42,
) -> SyntheticNodeClassificationData:
    """Create a graph where community membership is recoverable from neighborhoods."""
    generator = np.random.default_rng(seed)
    num_nodes = num_nodes_per_class * 2
    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[num_nodes_per_class:] = 1
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            same_block = labels[i] == labels[j]
            probability = p_in if same_block else p_out
            if generator.random() < probability:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    base_features = np.zeros((num_nodes, 2), dtype=np.float32)
    base_features[np.arange(num_nodes), labels] = 1.0
    noise = generator.normal(loc=0.0, scale=feature_noise, size=base_features.shape).astype(
        np.float32
    )
    features = base_features + noise

    indices = generator.permutation(num_nodes)
    train_cutoff = int(0.5 * num_nodes)
    val_cutoff = int(0.75 * num_nodes)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[indices[:train_cutoff]] = True
    val_mask[indices[train_cutoff:val_cutoff]] = True
    test_mask[indices[val_cutoff:]] = True

    return SyntheticNodeClassificationData(
        adjacency=torch.tensor(adjacency, dtype=torch.float32),
        features=torch.tensor(features, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
    )

