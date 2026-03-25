"""Synthetic PyG graph data mirroring the scratch GCN example."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.synthetic_graphs import generate_sbm_graph


def build_sbm_pyg_data(seed: int = 42, num_nodes_per_class: int = 40) -> Data:
    data = generate_sbm_graph(seed=seed, num_nodes_per_class=num_nodes_per_class)
    edge_index = torch.nonzero(data.adjacency > 0, as_tuple=False).t().contiguous()
    return Data(
        x=data.features,
        edge_index=edge_index,
        y=data.labels,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )

