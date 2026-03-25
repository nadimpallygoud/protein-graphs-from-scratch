"""Protein graph dataset helpers."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pdb_graphs import build_residue_graph_from_pdb, download_pdb_file
from utils.protein_graph import ProteinGraph
from utils.seed import DEFAULT_SEED


def load_metadata(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_dataset(
    metadata_rows: Sequence[Dict[str, str]],
    pdb_dir: Path,
    distance_threshold: float = 8.0,
) -> List[ProteinGraph]:
    dataset: List[ProteinGraph] = []
    for row in metadata_rows:
        pdb_path = download_pdb_file(row["pdb_id"], pdb_dir)
        graph = build_residue_graph_from_pdb(
            pdb_path=pdb_path,
            chain_id=row["chain_id"] or None,
            distance_threshold=distance_threshold,
            label=int(row["label"]),
            label_name=row["label_name"],
        )
        dataset.append(graph)
    return dataset


def split_dataset(
    dataset: Sequence[ProteinGraph],
    seed: int = DEFAULT_SEED,
    train_fraction: float = 0.67,
) -> Tuple[List[ProteinGraph], List[ProteinGraph]]:
    import random

    generator = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for index, graph in enumerate(dataset):
        label_to_indices.setdefault(int(graph.label), []).append(index)

    train_indices: List[int] = []
    test_indices: List[int] = []
    for indices in label_to_indices.values():
        generator.shuffle(indices)
        cutoff = max(1, int(len(indices) * train_fraction))
        if cutoff == len(indices) and len(indices) > 1:
            cutoff -= 1
        train_indices.extend(indices[:cutoff])
        test_indices.extend(indices[cutoff:])

    return [dataset[index] for index in train_indices], [dataset[index] for index in test_indices]
