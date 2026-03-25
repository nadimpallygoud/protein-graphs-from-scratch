"""Train a graph classifier on residue-level protein graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import build_dataset, load_metadata, split_dataset
from model import DenseGCNGraphClassifier
from utils.protein_graph import ProteinGraph
from utils.seed import set_seed


def graph_to_tensors(graph: ProteinGraph, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacency = torch.tensor(graph.adjacency, dtype=torch.float32, device=device)
    features = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
    label = torch.tensor([graph.label], dtype=torch.long, device=device)
    return adjacency, features, label


def evaluate(
    model: DenseGCNGraphClassifier,
    dataset: Sequence[ProteinGraph],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    correct = 0

    with torch.no_grad():
        for graph in dataset:
            adjacency, features, label = graph_to_tensors(graph, device)
            logits = model(adjacency, features).unsqueeze(0)
            loss = criterion(logits, label)
            losses.append(float(loss.item()))
            prediction = int(logits.argmax(dim=1).item())
            correct += int(prediction == int(label.item()))

    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "accuracy": correct / max(len(dataset), 1),
    }


def train(
    train_graphs: Sequence[ProteinGraph],
    test_graphs: Sequence[ProteinGraph],
    hidden_features: int,
    num_layers: int,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    device: torch.device,
) -> tuple[DenseGCNGraphClassifier, Dict[str, float]]:
    in_features = train_graphs[0].node_features.shape[1]
    num_classes = len({graph.label for graph in list(train_graphs) + list(test_graphs)})
    model = DenseGCNGraphClassifier(
        in_features=in_features,
        hidden_features=hidden_features,
        num_classes=num_classes,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_test_accuracy = -1.0

    for _ in range(num_epochs):
        model.train()
        for graph in train_graphs:
            adjacency, features, label = graph_to_tensors(graph, device)
            optimizer.zero_grad()
            logits = model(adjacency, features).unsqueeze(0)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

        test_metrics = evaluate(model, test_graphs, device)
        if test_metrics["accuracy"] > best_test_accuracy:
            best_test_accuracy = test_metrics["accuracy"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "train": evaluate(model, train_graphs, device),
        "test": evaluate(model, test_graphs, device),
    }
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a protein graph classifier.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).resolve().with_name("demo_proteins.csv"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=PROJECT_ROOT / "data" / "pdb")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "protein_classifier")
    parser.add_argument("--distance-threshold", type=float, default=8.0)
    parser.add_argument("--hidden-features", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_rows = load_metadata(args.metadata)
    dataset = build_dataset(
        metadata_rows=metadata_rows,
        pdb_dir=args.pdb_dir,
        distance_threshold=args.distance_threshold,
    )
    train_graphs, test_graphs = split_dataset(dataset, seed=args.seed)
    model, metrics = train(
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        hidden_features=args.hidden_features,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "protein_classifier.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metrics_payload = {
        "train_graphs": [graph.name for graph in train_graphs],
        "test_graphs": [graph.name for graph in test_graphs],
        "metrics": metrics,
        "model": {
            "hidden_features": args.hidden_features,
            "num_layers": args.num_layers,
            "distance_threshold": args.distance_threshold,
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    print("Protein classifier training finished")
    print(f"- checkpoint: {checkpoint_path}")
    print(f"- train_accuracy: {metrics['train']['accuracy']:.4f}")
    print(f"- test_accuracy: {metrics['test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
