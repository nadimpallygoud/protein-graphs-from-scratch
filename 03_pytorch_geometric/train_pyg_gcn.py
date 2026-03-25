"""Train a PyG GCN on the same synthetic node-classification graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.seed import set_seed
from dataset import build_sbm_pyg_data
from model import PyGNodeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyG GCN on a synthetic graph.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-features", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    return parser.parse_args()


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions[mask] == labels[mask]).float().mean().item())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = build_sbm_pyg_data(seed=args.seed)
    model = PyGNodeClassifier(
        in_features=data.x.size(1),
        hidden_features=args.hidden_features,
        num_classes=int(data.y.max().item()) + 1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val_accuracy = -1.0

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_accuracy = masked_accuracy(logits, data.y, data.val_mask)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    print("PyG GCN metrics")
    print(f"- train_accuracy: {masked_accuracy(logits, data.y, data.train_mask):.4f}")
    print(f"- val_accuracy: {masked_accuracy(logits, data.y, data.val_mask):.4f}")
    print(f"- test_accuracy: {masked_accuracy(logits, data.y, data.test_mask):.4f}")


if __name__ == "__main__":
    main()

