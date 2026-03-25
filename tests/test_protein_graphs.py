from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.pdb_graphs import build_residue_graph_from_pdb


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_build_residue_graph_from_fixture_pdb() -> None:
    graph = build_residue_graph_from_pdb(FIXTURE_DIR / "mini_protein.pdb", chain_id="A", distance_threshold=8.0)

    assert graph.num_nodes() == 3
    assert graph.residue_names == ["ALA", "GLU", "LYS"]
    assert graph.node_features.shape == (3, 25)
    assert graph.adjacency[0, 1] == 1.0
    assert graph.adjacency[1, 2] == 0.0


def test_point_mutation_changes_features_but_not_geometry() -> None:
    graph = build_residue_graph_from_pdb(FIXTURE_DIR / "mini_protein.pdb", chain_id="A", distance_threshold=9.0)
    mutated = graph.mutate_residue(chain_id="A", residue_number=2, new_residue_name="LYS")

    assert mutated.residue_names[1] == "LYS"
    assert np.array_equal(mutated.adjacency, graph.adjacency)
    assert np.array_equal(mutated.coordinates, graph.coordinates)
    assert not np.array_equal(mutated.node_features[1], graph.node_features[1])

