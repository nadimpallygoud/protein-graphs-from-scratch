"""Thin wrapper around the shared residue-graph builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pdb_graphs import build_residue_graph_from_pdb, graph_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a residue graph from a PDB file.")
    parser.add_argument("pdb_path", type=Path)
    parser.add_argument("--chain-id", type=str, default=None)
    parser.add_argument("--distance-threshold", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_residue_graph_from_pdb(
        pdb_path=args.pdb_path,
        chain_id=args.chain_id,
        distance_threshold=args.distance_threshold,
    )
    print(graph_summary(graph))
    print("First 10 residue labels:")
    for label in graph.residue_labels()[:10]:
        print(f"- {label}")


if __name__ == "__main__":
    main()
