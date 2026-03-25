"""Amino-acid encodings and simple physicochemical descriptors."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

AMINO_ACIDS: List[str] = list("ARNDCQEGHILKMFPSTWYV")
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
PROPERTY_NAMES = [
    "hydrophobicity",
    "charge",
    "polarity",
    "aromatic",
    "size",
]

# Normalized educational descriptors. The values are intentionally low-dimensional
# so the reader can inspect them, reason about them, and mutate them by hand.
AA_PROPERTIES: Dict[str, np.ndarray] = {
    "A": np.array([0.62, 0.0, 0.0, 0.0, 0.24], dtype=np.float32),
    "R": np.array([-2.53, 1.0, 1.0, 0.0, 0.84], dtype=np.float32),
    "N": np.array([-0.78, 0.0, 1.0, 0.0, 0.52], dtype=np.float32),
    "D": np.array([-0.90, -1.0, 1.0, 0.0, 0.50], dtype=np.float32),
    "C": np.array([0.29, 0.0, 0.0, 0.0, 0.41], dtype=np.float32),
    "Q": np.array([-0.85, 0.0, 1.0, 0.0, 0.60], dtype=np.float32),
    "E": np.array([-0.74, -1.0, 1.0, 0.0, 0.62], dtype=np.float32),
    "G": np.array([0.48, 0.0, 0.0, 0.0, 0.08], dtype=np.float32),
    "H": np.array([-0.40, 0.5, 1.0, 1.0, 0.66], dtype=np.float32),
    "I": np.array([1.38, 0.0, 0.0, 0.0, 0.73], dtype=np.float32),
    "L": np.array([1.06, 0.0, 0.0, 0.0, 0.73], dtype=np.float32),
    "K": np.array([-1.50, 1.0, 1.0, 0.0, 0.73], dtype=np.float32),
    "M": np.array([0.64, 0.0, 0.0, 0.0, 0.70], dtype=np.float32),
    "F": np.array([1.19, 0.0, 0.0, 1.0, 0.78], dtype=np.float32),
    "P": np.array([0.12, 0.0, 0.0, 0.0, 0.56], dtype=np.float32),
    "S": np.array([-0.18, 0.0, 1.0, 0.0, 0.39], dtype=np.float32),
    "T": np.array([-0.05, 0.0, 1.0, 0.0, 0.51], dtype=np.float32),
    "W": np.array([0.81, 0.0, 0.0, 1.0, 1.00], dtype=np.float32),
    "Y": np.array([0.26, 0.0, 1.0, 1.0, 0.87], dtype=np.float32),
    "V": np.array([1.08, 0.0, 0.0, 0.0, 0.60], dtype=np.float32),
}


def is_standard_amino_acid(residue_name: str) -> bool:
    return residue_name.upper() in THREE_TO_ONE


def three_to_one(residue_name: str) -> str:
    residue_name = residue_name.upper()
    if residue_name not in THREE_TO_ONE:
        raise KeyError(f"Unsupported residue name: {residue_name}")
    return THREE_TO_ONE[residue_name]


def one_hot_encode_residue(residue_name: str) -> np.ndarray:
    one_letter = three_to_one(residue_name)
    vector = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    vector[AA_TO_INDEX[one_letter]] = 1.0
    return vector


def physicochemical_vector(residue_name: str) -> np.ndarray:
    return AA_PROPERTIES[three_to_one(residue_name)].copy()


def residue_feature_vector(residue_name: str) -> np.ndarray:
    return np.concatenate(
        [one_hot_encode_residue(residue_name), physicochemical_vector(residue_name)],
        axis=0,
    )

