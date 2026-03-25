"""Minimal relevant-walk demo inspired by public GNN-LRP examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dense_gcn import DenseGCNGraphClassifier
from utils.seed import set_seed
from utils.visualization import plot_graph_importance
from gnn_lrp import explain_graph_with_gnn_lrp
from relevant_walks import extract_relevant_walks


def make_toy_graph(has_triangle: bool, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    generator = np.random.default_rng(seed)
    adjacency = np.zeros((6, 6), dtype=np.float32)
    chain_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for source, target in chain_edges:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    if has_triangle:
        motif_edges = [(1, 3), (1, 2), (2, 3)]
        for source, target in motif_edges:
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
    if generator.random() < 0.3:
        adjacency[0, 2] = adjacency[2, 0] = 1.0
    features = np.ones((6, 2), dtype=np.float32)
    features[:, 1] = np.linspace(0.0, 1.0, 6)
    return adjacency, features, int(has_triangle)


def build_dataset(num_graphs: int = 40, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    dataset = []
    for index in range(num_graphs):
        has_triangle = index % 2 == 0
        dataset.append(make_toy_graph(has_triangle=has_triangle, seed=seed + index))
    return dataset


def train_demo_model(dataset: List[Tuple[np.ndarray, np.ndarray, int]]) -> DenseGCNGraphClassifier:
    model = DenseGCNGraphClassifier(in_features=2, hidden_features=16, num_classes=2, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(120):
        model.train()
        for adjacency, features, label in dataset:
            optimizer.zero_grad()
            logits = model(
                torch.tensor(adjacency, dtype=torch.float32),
                torch.tensor(features, dtype=torch.float32),
            ).unsqueeze(0)
            loss = criterion(logits, torch.tensor([label], dtype=torch.long))
            loss.backward()
            optimizer.step()
    return model


def main() -> None:
    set_seed(42)
    dataset = build_dataset()
    model = train_demo_model(dataset)
    adjacency, features, _ = dataset[0]
    explanation = explain_graph_with_gnn_lrp(model, adjacency=adjacency, node_features=features)
    walks = extract_relevant_walks(
        adjacency=adjacency,
        node_relevance=explanation.node_relevance,
        edge_relevance=explanation.edge_relevance,
        max_length=3,
        top_k=5,
    )

    output_dir = PROJECT_ROOT / "artifacts" / "gnn_lrp_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_graph_importance(
        adjacency=adjacency,
        node_importance=explanation.node_relevance,
        edge_importance=explanation.edge_relevance,
        title="Toy Relevant-Walk Demo",
        output_path=output_dir / "toy_relevance.png",
    )
    payload = {
        "class_probability": explanation.class_probability,
        "node_relevance": explanation.node_relevance.tolist(),
        "walks": [{"nodes": walk.nodes, "score": walk.score} for walk in walks],
    }
    (output_dir / "walks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved demo outputs to {output_dir}")


if __name__ == "__main__":
    main()

