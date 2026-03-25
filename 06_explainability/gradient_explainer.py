"""Baseline explainers for dense protein graph classifiers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dense_gcn import DenseGCNGraphClassifier


def _target_class_from_logits(logits: torch.Tensor, target_class: Optional[int]) -> int:
    return int(logits.argmax().item()) if target_class is None else int(target_class)


def node_gradient_importance(
    model: DenseGCNGraphClassifier,
    adjacency: np.ndarray,
    node_features: np.ndarray,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.zero_grad(set_to_none=True)
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    features = torch.tensor(node_features, dtype=torch.float32, requires_grad=True)
    logits = model(adjacency_tensor, features)
    class_index = _target_class_from_logits(logits, target_class)
    logits[class_index].backward()
    gradients = features.grad.detach()
    feature_importance = (gradients * features.detach()).abs()
    node_importance = feature_importance.sum(dim=1)
    model.zero_grad(set_to_none=True)
    return node_importance.cpu().numpy(), feature_importance.cpu().numpy()


def node_occlusion_importance(
    model: DenseGCNGraphClassifier,
    adjacency: np.ndarray,
    node_features: np.ndarray,
    target_class: Optional[int] = None,
) -> np.ndarray:
    model.eval()
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    features = torch.tensor(node_features, dtype=torch.float32)
    with torch.no_grad():
        baseline_logits = model(adjacency_tensor, features)
        class_index = _target_class_from_logits(baseline_logits, target_class)
        baseline_score = float(baseline_logits[class_index].item())

    importance = np.zeros(node_features.shape[0], dtype=np.float32)
    for node_index in range(node_features.shape[0]):
        occluded = features.clone()
        occluded[node_index] = 0.0
        with torch.no_grad():
            score = float(model(adjacency_tensor, occluded)[class_index].item())
        importance[node_index] = baseline_score - score
    return importance


def edge_occlusion_importance(
    model: DenseGCNGraphClassifier,
    adjacency: np.ndarray,
    node_features: np.ndarray,
    target_class: Optional[int] = None,
) -> np.ndarray:
    model.eval()
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    features = torch.tensor(node_features, dtype=torch.float32)
    with torch.no_grad():
        baseline_logits = model(adjacency_tensor, features)
        class_index = _target_class_from_logits(baseline_logits, target_class)
        baseline_score = float(baseline_logits[class_index].item())

    importance = np.zeros_like(adjacency, dtype=np.float32)
    edge_indices = np.argwhere(np.triu(adjacency, k=1) > 0)
    for source, target in edge_indices:
        occluded = adjacency_tensor.clone()
        occluded[source, target] = 0.0
        occluded[target, source] = 0.0
        with torch.no_grad():
            score = float(model(occluded, features)[class_index].item())
        importance[source, target] = baseline_score - score
        importance[target, source] = importance[source, target]
    return importance
