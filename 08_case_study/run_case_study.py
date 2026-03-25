"""Run biological case studies with gradient and GNN-LRP explanations."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


# Allow this script to be run from either the repository root or the 08_case_study directory
# without encountering ModuleNotFoundError for the local package imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "06_explainability"))
sys.path.insert(0, str(PROJECT_ROOT / "07_gnn_lrp"))

from gradient_explainer import node_gradient_importance
from gnn_lrp import explain_graph_with_gnn_lrp
from relevant_walks import extract_relevant_walks
from utils.dense_gcn import DenseGCNGraphClassifier
from utils.pdb_graphs import build_residue_graph_from_pdb, download_pdb_file
from utils.visualization import plot_graph_importance, plot_residue_importance, write_relevance_pymol_script


DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "protein_classifier" / "protein_classifier.pt"
DEFAULT_CASES = Path(__file__).resolve().with_name("case_studies.json")


def load_case(case_name: str, cases_path: Path) -> Dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if case_name not in cases:
        raise KeyError(f"Unknown case study '{case_name}'. Available: {sorted(cases)}")
    return cases[case_name]


def load_model(checkpoint_path: Path, in_features: int, hidden_features: int, num_layers: int) -> DenseGCNGraphClassifier:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run 05_training/train_protein_classifier.py first."
        )
    model = DenseGCNGraphClassifier(
        in_features=in_features,
        hidden_features=hidden_features,
        num_classes=2,
        num_layers=num_layers,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def top_residues(residue_labels: list[str], scores: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
    order = np.argsort(scores)[::-1][:top_k]
    return [{"residue": residue_labels[index], "score": float(scores[index])} for index in order]


def focused_subgraph(
    graph,
    node_scores: np.ndarray,
    edge_scores: np.ndarray,
    max_nodes: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    selected = np.argsort(node_scores)[::-1][: min(max_nodes, graph.num_nodes())]
    adjacency = graph.adjacency[np.ix_(selected, selected)]
    node_relevance = node_scores[selected]
    edge_relevance = edge_scores[np.ix_(selected, selected)]
    labels = [graph.residue_labels()[index] for index in selected]
    return adjacency, node_relevance, edge_relevance, labels


def explain_single_graph(model: DenseGCNGraphClassifier, graph, output_prefix: Path) -> Dict[str, Any]:
    gradient_scores, _ = node_gradient_importance(model, graph.adjacency, graph.node_features)
    lrp_result = explain_graph_with_gnn_lrp(model, graph.adjacency, graph.node_features)
    walk_adjacency, walk_nodes, walk_edges, walk_labels = focused_subgraph(
        graph=graph,
        node_scores=lrp_result.node_relevance,
        edge_scores=lrp_result.edge_relevance,
        max_nodes=25,
    )
    walks = extract_relevant_walks(
        adjacency=walk_adjacency,
        node_relevance=walk_nodes,
        edge_relevance=walk_edges,
        max_length=3,
        top_k=10,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_adjacency, plot_nodes, plot_edges, plot_labels = focused_subgraph(
        graph=graph,
        node_scores=lrp_result.node_relevance,
        edge_scores=lrp_result.edge_relevance,
        max_nodes=40,
    )
    plot_graph_importance(
        adjacency=plot_adjacency,
        node_labels=plot_labels,
        node_importance=plot_nodes,
        edge_importance=plot_edges,
        title=f"{graph.name} GNN-LRP (focused subgraph)",
        output_path=output_prefix.with_suffix(".graph.png"),
    )
    plot_residue_importance(
        residue_labels=graph.residue_labels(),
        relevance=lrp_result.node_relevance,
        title=f"{graph.name} Residue Relevance",
        top_k=20,
        output_path=output_prefix.with_suffix(".residues.png"),
    )

    return {
        "graph_name": graph.name,
        "class_index": lrp_result.class_index,
        "class_probability": lrp_result.class_probability,
        "top_gradient_residues": top_residues(graph.residue_labels(), gradient_scores),
        "top_lrp_residues": top_residues(graph.residue_labels(), lrp_result.node_relevance),
        "node_relevance": lrp_result.node_relevance.tolist(),
        "edge_relevance": lrp_result.edge_relevance.tolist(),
        "relevant_walks": [
            {"residues": [walk_labels[index] for index in walk.nodes], "score": walk.score}
            for walk in walks
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a protein explanation case study.")
    parser.add_argument("--case-name", type=str, default="lysozyme_active_site")
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pdb-dir", type=Path, default=PROJECT_ROOT / "data" / "pdb")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "case_studies")
    parser.add_argument("--distance-threshold", type=float, default=8.0)
    parser.add_argument("--hidden-features", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = load_case(args.case_name, args.cases_path)
    pdb_path = download_pdb_file(case["pdb_id"], args.pdb_dir)
    base_graph = build_residue_graph_from_pdb(
        pdb_path=pdb_path,
        chain_id=case["chain_id"],
        distance_threshold=args.distance_threshold,
        label=case["label"],
        label_name=case["label_name"],
    )
    model = load_model(
        checkpoint_path=args.checkpoint,
        in_features=base_graph.node_features.shape[1],
        hidden_features=args.hidden_features,
        num_layers=args.num_layers,
    )

    case_output_dir = args.output_dir / args.case_name
    base_payload = explain_single_graph(model, base_graph, case_output_dir / "wild_type")
    payload: Dict[str, Any] = {
        "case_name": args.case_name,
        "description": case["description"],
        "known_residues": case.get("known_residues", []),
        "wild_type": base_payload,
    }

    write_relevance_pymol_script(
        residue_labels=base_graph.residue_labels(),
        relevance=np.array(base_payload["node_relevance"], dtype=np.float32),
        pdb_path=pdb_path,
        script_path=case_output_dir / "wild_type.pml",
    )

    if "mutation" in case:
        mutation = case["mutation"]
        mutated_graph = base_graph.mutate_residue(
            chain_id=mutation["chain_id"],
            residue_number=int(mutation["residue_number"]),
            new_residue_name=mutation["to"],
        )
        mutated_payload = explain_single_graph(model, mutated_graph, case_output_dir / "mutant")
        payload["mutation"] = mutation
        payload["mutant"] = mutated_payload
        delta = (
            np.array(mutated_payload["node_relevance"], dtype=np.float32)
            - np.array(base_payload["node_relevance"], dtype=np.float32)
        )
        payload["delta_top_residues"] = top_residues(base_graph.residue_labels(), np.abs(delta))
        write_relevance_pymol_script(
            residue_labels=mutated_graph.residue_labels(),
            relevance=np.array(mutated_payload["node_relevance"], dtype=np.float32),
            pdb_path=pdb_path,
            script_path=case_output_dir / "mutant.pml",
        )

    case_output_dir.mkdir(parents=True, exist_ok=True)
    (case_output_dir / "explanations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved case-study outputs to {case_output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Case-study run failed: {error}", file=sys.stderr)
        traceback.print_exc()
        raise
