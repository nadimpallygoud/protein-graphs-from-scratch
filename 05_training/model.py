"""Training-facing wrappers around the shared dense GCN model."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dense_gcn import DenseGCNGraphClassifier, normalize_adjacency_torch


__all__ = ["DenseGCNGraphClassifier", "normalize_adjacency_torch"]

