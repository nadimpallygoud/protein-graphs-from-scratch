"""Dense GCN implementation from first principles."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


def normalize_adjacency_torch(adjacency: Tensor, add_self_loops: bool = True) -> Tensor:
    if add_self_loops:
        adjacency = adjacency + torch.eye(adjacency.size(0), device=adjacency.device)
    degree = adjacency.sum(dim=1)
    inv_sqrt_degree = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
    return inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]


class DenseGCNLayer(nn.Module):
    """A dense educational GCN layer that exposes the propagation rule directly."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, adjacency: Tensor, features: Tensor) -> Tensor:
        normalized_adjacency = normalize_adjacency_torch(adjacency)
        return normalized_adjacency @ self.linear(features)


class ScratchGCN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int) -> None:
        super().__init__()
        self.gcn1 = DenseGCNLayer(in_features, hidden_features)
        self.gcn2 = DenseGCNLayer(hidden_features, num_classes)
        self.activation = nn.ReLU()

    def forward(self, adjacency: Tensor, features: Tensor) -> Tensor:
        hidden = self.activation(self.gcn1(adjacency, features))
        return self.gcn2(adjacency, hidden)


def accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    predictions = logits.argmax(dim=1)
    correct = (predictions[mask] == labels[mask]).float().mean()
    return float(correct.item())


def train_scratch_model(
    adjacency: Tensor,
    features: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    hidden_features: int = 16,
    learning_rate: float = 1e-2,
    weight_decay: float = 5e-4,
    num_epochs: int = 200,
) -> Dict[str, float]:
    model = ScratchGCN(
        in_features=features.size(1),
        hidden_features=hidden_features,
        num_classes=int(labels.max().item()) + 1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = -1.0
    best_state = None

    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(adjacency, features)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(adjacency, features)
            val_accuracy = accuracy(logits, labels, val_mask)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(adjacency, features)
    return {
        "train_accuracy": accuracy(logits, labels, train_mask),
        "val_accuracy": accuracy(logits, labels, val_mask),
        "test_accuracy": accuracy(logits, labels, test_mask),
    }

