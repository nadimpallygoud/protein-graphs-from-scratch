from __future__ import annotations

import numpy as np

from utils.dense_gcn import DenseGCNGraphClassifier
from utils.seed import set_seed
from utils.synthetic_graphs import generate_sbm_graph
from helpers import load_module


gradient_explainer = load_module(
    "gradient_explainer_module",
    "06_explainability/gradient_explainer.py",
)
gnn_lrp = load_module("gnn_lrp_module", "07_gnn_lrp/gnn_lrp.py")
relevant_walks = load_module("relevant_walks_module", "07_gnn_lrp/relevant_walks.py")


def test_gnn_lrp_preserves_relevance_mass() -> None:
    set_seed(7)
    data = generate_sbm_graph(num_nodes_per_class=6, seed=7)
    model = DenseGCNGraphClassifier(in_features=2, hidden_features=8, num_classes=2, num_layers=2)

    explanation = gnn_lrp.explain_graph_with_gnn_lrp(
        model=model,
        adjacency=data.adjacency.numpy(),
        node_features=data.features.numpy(),
    )

    assert np.isfinite(explanation.node_relevance).all()
    assert np.isfinite(explanation.edge_relevance).all()
    assert np.isfinite(explanation.feature_relevance).all()
    assert abs(float(explanation.node_relevance.sum()) - explanation.class_probability) < 1e-4


def test_gradient_then_lrp_runs_on_same_model_without_failure() -> None:
    set_seed(11)
    data = generate_sbm_graph(num_nodes_per_class=12, seed=11)
    model = DenseGCNGraphClassifier(in_features=2, hidden_features=8, num_classes=2, num_layers=2)

    gradient_scores, _ = gradient_explainer.node_gradient_importance(
        model=model,
        adjacency=data.adjacency.numpy(),
        node_features=data.features.numpy(),
    )
    explanation = gnn_lrp.explain_graph_with_gnn_lrp(
        model=model,
        adjacency=data.adjacency.numpy(),
        node_features=data.features.numpy(),
    )

    assert gradient_scores.shape[0] == data.features.shape[0]
    assert explanation.node_relevance.shape[0] == data.features.shape[0]


def test_relevant_walk_extraction_returns_nonempty_ranked_walks() -> None:
    set_seed(5)
    data = generate_sbm_graph(num_nodes_per_class=5, seed=5)
    model = DenseGCNGraphClassifier(in_features=2, hidden_features=8, num_classes=2, num_layers=2)
    explanation = gnn_lrp.explain_graph_with_gnn_lrp(
        model=model,
        adjacency=data.adjacency.numpy(),
        node_features=data.features.numpy(),
    )
    walks = relevant_walks.extract_relevant_walks(
        adjacency=data.adjacency.numpy(),
        node_relevance=explanation.node_relevance,
        edge_relevance=explanation.edge_relevance,
        max_length=2,
        top_k=5,
    )

    assert walks
    assert walks[0].score >= walks[-1].score
