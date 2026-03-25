"""Train the scratch GCN on a synthetic node-classification graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.seed import set_seed
from utils.synthetic_graphs import generate_sbm_graph
from models import train_scratch_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the scratch GCN on an SBM graph.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes-per-class", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = generate_sbm_graph(num_nodes_per_class=args.num_nodes_per_class, seed=args.seed)
    metrics = train_scratch_model(
        adjacency=data.adjacency,
        features=data.features,
        labels=data.labels,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_epochs=args.epochs,
    )
    print("Scratch GCN metrics")
    for metric_name, value in metrics.items():
        print(f"- {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()

