"""Rank short relevant walks from node and edge relevance scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


@dataclass
class RelevantWalk:
    nodes: Tuple[int, ...]
    score: float


def _canonical_path(path: Sequence[int]) -> Tuple[int, ...]:
    forward = tuple(path)
    backward = tuple(reversed(path))
    return min(forward, backward)


def _walk_score(path: Sequence[int], node_relevance: np.ndarray, edge_relevance: np.ndarray) -> float:
    score = float(np.sum(node_relevance[list(path)]))
    for source, target in zip(path[:-1], path[1:]):
        score += float(edge_relevance[source, target])
    return score


def extract_relevant_walks(
    adjacency: np.ndarray,
    node_relevance: np.ndarray,
    edge_relevance: np.ndarray,
    max_length: int = 3,
    top_k: int = 10,
) -> List[RelevantWalk]:
    graph = adjacency > 0
    seen: Set[Tuple[int, ...]] = set()
    walks: List[RelevantWalk] = []
    seed_nodes = np.argsort(node_relevance)[::-1][: min(10, adjacency.shape[0])]

    def dfs(path: List[int]) -> None:
        canonical = _canonical_path(path)
        if len(path) > 1 and canonical not in seen:
            seen.add(canonical)
            walks.append(
                RelevantWalk(
                    nodes=canonical,
                    score=_walk_score(canonical, node_relevance=node_relevance, edge_relevance=edge_relevance),
                )
            )
        if len(path) - 1 == max_length:
            return
        current = path[-1]
        for neighbor in np.where(graph[current])[0].tolist():
            if neighbor in path:
                continue
            dfs(path + [neighbor])

    for seed in seed_nodes.tolist():
        dfs([seed])

    walks.sort(key=lambda walk: walk.score, reverse=True)
    return walks[:top_k]

