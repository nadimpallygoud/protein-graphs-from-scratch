"""Shared helpers for downloading PDB files and building residue graphs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
from Bio.PDB import PDBParser

from utils.aa_properties import is_standard_amino_acid, residue_feature_vector
from utils.protein_graph import ProteinGraph


def download_pdb_file(pdb_id: str, output_dir: Path) -> Path:
    pdb_id = pdb_id.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdb_id}.pdb"
    if output_path.exists():
        return output_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    urlretrieve(url, output_path)
    return output_path


def residue_coordinate(residue) -> np.ndarray:
    if "CA" in residue:
        return residue["CA"].coord.astype(np.float32)
    coordinates = np.array([atom.coord for atom in residue.get_atoms()], dtype=np.float32)
    return coordinates.mean(axis=0).astype(np.float32)


def extract_standard_residues(
    pdb_path: Path,
    chain_id: Optional[str] = None,
) -> List[Tuple[str, int, str, np.ndarray]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    residues: List[Tuple[str, int, str, np.ndarray]] = []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain:
            hetfield, residue_number, _ = residue.id
            if hetfield.strip():
                continue
            residue_name = residue.resname.upper()
            if not is_standard_amino_acid(residue_name):
                continue
            coordinates = residue_coordinate(residue)
            residues.append((chain.id, int(residue_number), residue_name, coordinates))
    return residues


def build_residue_graph_from_pdb(
    pdb_path: Path,
    chain_id: Optional[str] = None,
    distance_threshold: float = 8.0,
    label: Optional[int] = None,
    label_name: Optional[str] = None,
) -> ProteinGraph:
    residues = extract_standard_residues(pdb_path=pdb_path, chain_id=chain_id)
    if not residues:
        raise ValueError(f"No standard amino-acid residues found in {pdb_path}")

    coordinates = np.stack([coordinate for _, _, _, coordinate in residues], axis=0)
    residue_names = [name for _, _, name, _ in residues]
    chain_ids = [chain for chain, _, _, _ in residues]
    residue_numbers = [number for _, number, _, _ in residues]
    node_features = np.stack([residue_feature_vector(name) for name in residue_names], axis=0)

    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    adjacency = (distances <= distance_threshold).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)

    graph_name = pdb_path.stem if chain_id is None else f"{pdb_path.stem}_{chain_id}"
    return ProteinGraph(
        name=graph_name,
        adjacency=adjacency,
        node_features=node_features,
        coordinates=coordinates,
        residue_names=residue_names,
        chain_ids=chain_ids,
        residue_numbers=residue_numbers,
        label=label,
        label_name=label_name,
    )


def graph_summary(graph: ProteinGraph) -> str:
    num_edges = int(np.sum(graph.adjacency) // 2)
    return (
        f"ProteinGraph(name={graph.name}, num_nodes={graph.num_nodes()}, num_edges={num_edges}, "
        f"label={graph.label_name})"
    )

