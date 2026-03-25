"""Matrix-based graph basics used throughout the repository."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


Edge = Tuple[int, int]


def build_adjacency_matrix(
    num_nodes: int,
    edges: Sequence[Edge],
    undirected: bool = True,
    add_self_loops: bool = False,
) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for source, target in edges:
        adjacency[source, target] = 1.0
        if undirected:
            adjacency[target, source] = 1.0
    if add_self_loops:
        adjacency += np.eye(num_nodes, dtype=np.float32)
    return adjacency


def degree_matrix(adjacency: np.ndarray) -> np.ndarray:
    degrees = np.sum(adjacency, axis=1)
    return np.diag(degrees.astype(np.float32))


def laplacian(adjacency: np.ndarray, normalized: bool = False) -> np.ndarray:
    degree = degree_matrix(adjacency)
    if not normalized:
        return degree - adjacency

    diagonal = np.diag(degree)
    inv_sqrt_degree = np.zeros_like(diagonal)
    valid = diagonal > 0
    inv_sqrt_degree[valid] = 1.0 / np.sqrt(diagonal[valid])
    normalized_adjacency = (inv_sqrt_degree[:, None] * adjacency) * inv_sqrt_degree[None, :]
    identity = np.eye(adjacency.shape[0], dtype=np.float32)
    return identity - normalized_adjacency


def normalize_adjacency(adjacency: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    working_adjacency = adjacency.astype(np.float32).copy()
    if add_self_loops:
        working_adjacency += np.eye(adjacency.shape[0], dtype=np.float32)
    degree = np.sum(working_adjacency, axis=1)
    inv_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    return (inv_sqrt_degree[:, None] * working_adjacency) * inv_sqrt_degree[None, :]


def neighbors(adjacency: np.ndarray, node_index: int) -> List[int]:
    return np.where(adjacency[node_index] > 0)[0].astype(int).tolist()


def k_hop_reachability(adjacency: np.ndarray, source: int, num_hops: int) -> List[int]:
    reachability = np.zeros(adjacency.shape[0], dtype=np.float32)
    reachability[source] = 1.0
    current = reachability.copy()
    for _ in range(num_hops):
        current = adjacency @ current
        reachability = np.maximum(reachability, (current > 0).astype(np.float32))
    return np.where(reachability > 0)[0].astype(int).tolist()


def demo_graph() -> Tuple[np.ndarray, List[str]]:
    labels = ["A", "B", "C", "D", "E"]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)]
    adjacency = build_adjacency_matrix(num_nodes=len(labels), edges=edges, undirected=True)
    return adjacency, labels


def pretty_print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name}:")
    print(np.array2string(matrix, precision=3, suppress_small=True))
    print()


def main() -> None:
    adjacency, labels = demo_graph()
    print(f"Labels: {labels}\n")
    pretty_print_matrix("Adjacency", adjacency)
    pretty_print_matrix("Degree", degree_matrix(adjacency))
    pretty_print_matrix("Laplacian", laplacian(adjacency))
    pretty_print_matrix("Normalized Laplacian", laplacian(adjacency, normalized=True))
    pretty_print_matrix("Normalized adjacency", normalize_adjacency(adjacency))
    print(f"Neighbors of node B: {neighbors(adjacency, 1)}")
    print(f"2-hop reachability from node A: {k_hop_reachability(adjacency, 0, 2)}")


if __name__ == "__main__":
    main()

