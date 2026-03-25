"""Plotting helpers for graph and residue relevance."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph_importance(
    adjacency: np.ndarray,
    node_labels: Optional[Iterable[str]] = None,
    node_importance: Optional[np.ndarray] = None,
    edge_importance: Optional[np.ndarray] = None,
    title: str = "Graph Importance",
    output_path: Optional[Path] = None,
) -> None:
    """Use a spring layout because proteins rarely have clean planar projections."""
    graph = nx.from_numpy_array(adjacency)
    layout = nx.spring_layout(graph, seed=42)
    node_importance = (
        np.asarray(node_importance, dtype=np.float32)
        if node_importance is not None
        else np.ones(adjacency.shape[0], dtype=np.float32)
    )
    labels = (
        {index: label for index, label in enumerate(node_labels)}
        if node_labels is not None
        else None
    )
    if labels is not None and adjacency.shape[0] > 60:
        # Dense residue labels make large protein plots unreadable and can overwhelm headless renders.
        labels = None
    edge_widths = None
    if edge_importance is not None:
        widths = []
        for u, v in graph.edges():
            widths.append(1.0 + 5.0 * float(edge_importance[u, v]))
        edge_widths = widths

    plt.figure(figsize=(9, 7))
    nx.draw_networkx(
        graph,
        pos=layout,
        labels=labels,
        node_color=node_importance,
        cmap="YlOrRd",
        width=edge_widths if edge_widths is not None else 1.5,
        edge_color="#6B7280",
        node_size=220 if adjacency.shape[0] > 60 else 500,
        font_size=7,
    )
    plt.title(title)
    plt.axis("off")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_residue_importance(
    residue_labels: Iterable[str],
    relevance: np.ndarray,
    title: str = "Residue Relevance",
    top_k: int = 20,
    output_path: Optional[Path] = None,
) -> None:
    labels = list(residue_labels)
    relevance = np.asarray(relevance, dtype=np.float32)
    order = np.argsort(relevance)[::-1][:top_k]

    plt.figure(figsize=(10, 6))
    plt.bar(
        [labels[index] for index in order],
        relevance[order],
        color="#D97706",
    )
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Relevance")
    plt.title(title)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_relevance_pymol_script(
    residue_labels: Iterable[str],
    relevance: np.ndarray,
    pdb_path: Path,
    script_path: Path,
) -> None:
    """Write a tiny PyMOL script so the explanation can be inspected in 3D."""
    residue_labels = list(residue_labels)
    relevance = np.asarray(relevance, dtype=np.float32)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    max_relevance = float(np.max(relevance)) if np.max(relevance) > 0 else 1.0

    lines = [
        "reinitialize",
        f"load {pdb_path.as_posix()}, protein",
        "hide everything, protein",
        "show cartoon, protein",
        "spectrum b, blue_white_red, protein",
    ]
    for label, score in zip(residue_labels, relevance):
        chain, residue = label.split(":")
        residue_number = "".join(character for character in residue if character.isdigit())
        normalized = score / max_relevance
        lines.append(
            f"alter protein and chain {chain} and resi {residue_number}, b={normalized:.6f}"
        )
    lines.extend(
        [
            "rebuild",
            "spectrum b, blue_white_red, protein",
            "show sticks, byres (protein within 4 of (b > 0.6))",
            "bg_color white",
        ]
    )
    script_path.write_text("\n".join(lines), encoding="utf-8")
