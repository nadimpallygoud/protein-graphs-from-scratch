"""Thin wrapper around the shared PDB download helper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pdb_graphs import download_pdb_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a PDB file from RCSB.")
    parser.add_argument("pdb_id", type=str)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "pdb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = download_pdb_file(args.pdb_id, args.output_dir)
    print(output_path)


if __name__ == "__main__":
    main()
