"""Rebuild plots from a saved case-study explanation JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pdb_graphs import build_residue_graph_from_pdb, download_pdb_file
from utils.visualization import plot_graph_importance, plot_residue_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved case-study explanation.")
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--pdb-id", type=str, required=True)
    parser.add_argument("--chain-id", type=str, default="A")
    parser.add_argument("--pdb-dir", type=Path, default=PROJECT_ROOT / "data" / "pdb")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "case_studies" / "plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.case_json.read_text(encoding="utf-8"))
    pdb_path = download_pdb_file(args.pdb_id, args.pdb_dir)
    graph = build_residue_graph_from_pdb(pdb_path, chain_id=args.chain_id)
    wild_type = payload["wild_type"]
    node_relevance = np.array(wild_type["node_relevance"], dtype=np.float32)
    edge_relevance = np.array(wild_type["edge_relevance"], dtype=np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_graph_importance(
        adjacency=graph.adjacency,
        node_labels=graph.residue_labels(),
        node_importance=node_relevance,
        edge_importance=edge_relevance,
        title=f"{graph.name} Wild-Type Relevance",
        output_path=args.output_dir / "wild_type_graph.png",
    )
    plot_residue_importance(
        residue_labels=graph.residue_labels(),
        relevance=node_relevance,
        title=f"{graph.name} Wild-Type Residues",
        output_path=args.output_dir / "wild_type_residues.png",
    )
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()

