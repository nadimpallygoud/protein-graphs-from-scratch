from __future__ import annotations

import numpy as np

from helpers import load_module


graph_basics = load_module("graph_basics_module", "01_graph_basics/graph_basics.py")


def test_degree_and_laplacian_are_consistent() -> None:
    adjacency, _ = graph_basics.demo_graph()
    degree = graph_basics.degree_matrix(adjacency)
    laplacian = graph_basics.laplacian(adjacency)

    assert np.allclose(np.diag(degree), adjacency.sum(axis=1))
    assert np.allclose(laplacian, degree - adjacency)


def test_normalized_adjacency_is_symmetric_and_self_connected() -> None:
    adjacency, _ = graph_basics.demo_graph()
    normalized = graph_basics.normalize_adjacency(adjacency)

    assert normalized.shape == adjacency.shape
    assert np.allclose(normalized, normalized.T)
    assert np.all(np.diag(normalized) > 0.0)


def test_k_hop_reachability_captures_two_hop_neighbors() -> None:
    adjacency, labels = graph_basics.demo_graph()
    reachable = graph_basics.k_hop_reachability(adjacency, source=0, num_hops=2)

    reachable_labels = {labels[index] for index in reachable}
    assert {"A", "B", "C", "E"}.issubset(reachable_labels)
