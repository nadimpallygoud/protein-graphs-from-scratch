"""Shared protein graph container."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from utils.aa_properties import residue_feature_vector


@dataclass
class ProteinGraph:
    """Residue-level protein graph used by the training and explanation pipeline."""

    name: str
    adjacency: np.ndarray
    node_features: np.ndarray
    coordinates: np.ndarray
    residue_names: List[str]
    chain_ids: List[str]
    residue_numbers: List[int]
    label: Optional[int] = None
    label_name: Optional[str] = None

    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    def copy(self) -> "ProteinGraph":
        return replace(
            self,
            adjacency=self.adjacency.copy(),
            node_features=self.node_features.copy(),
            coordinates=self.coordinates.copy(),
            residue_names=list(self.residue_names),
            chain_ids=list(self.chain_ids),
            residue_numbers=list(self.residue_numbers),
        )

    def mutate_residue(
        self,
        chain_id: str,
        residue_number: int,
        new_residue_name: str,
    ) -> "ProteinGraph":
        """
        Change node features in place of a point mutation while preserving structure.
        
        NOTE: This is a purely *in silico* point mutation. It modifies the node 
        chemical identity (features) and label, but forces the 3D backbone and 
        sidechain coordinates to remain completely identical to the wild-type. 
        This is mathematically necessary to isolate the GNN-LRP signal delta caused 
        by the chemistry change alone, but it does NOT simulate real structural 
        relaxation or energy minimization.
        """
        mutated = self.copy()
        for index, (chain, number) in enumerate(zip(self.chain_ids, self.residue_numbers)):
            if chain == chain_id and number == residue_number:
                mutated.residue_names[index] = new_residue_name.upper()
                mutated.node_features[index] = residue_feature_vector(new_residue_name.upper())
                mutated.name = f"{self.name}_{chain_id}{residue_number}{new_residue_name.upper()}"
                return mutated
        raise ValueError(f"Residue {chain_id}:{residue_number} not found in graph {self.name}")

    def edge_index(self) -> np.ndarray:
        rows, cols = np.where(self.adjacency > 0)
        return np.stack([rows, cols], axis=0).astype(np.int64)

    def to_pyg_data(self):
        """Import lazily so the rest of the repository remains usable without PyG."""
        import torch
        from torch_geometric.data import Data

        return Data(
            x=torch.tensor(self.node_features, dtype=torch.float32),
            edge_index=torch.tensor(self.edge_index(), dtype=torch.long),
            y=None if self.label is None else torch.tensor([self.label], dtype=torch.long),
            pos=torch.tensor(self.coordinates, dtype=torch.float32),
        )

    def residue_labels(self) -> List[str]:
        return [
            f"{chain}:{name}{number}"
            for chain, name, number in zip(self.chain_ids, self.residue_names, self.residue_numbers)
        ]

