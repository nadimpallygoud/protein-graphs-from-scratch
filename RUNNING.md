# How to Run the Repository

This repository is intended to be run from the project root:

```bash
cd gnn-protein-lrp
```

## 1. Create an Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `torch-geometric` needs a platform-specific install, install PyTorch first and then follow the official wheel instructions for your CUDA or CPU setup.

## 2. Graph Basics

```bash
python 01_graph_basics/graph_basics.py
```

This prints:

- adjacency
- degree matrix
- Laplacian
- normalized adjacency

## 3. Scratch GCN

```bash
python 02_gnn_from_scratch/train_scratch_gcn.py --epochs 200
```

This trains a small dense GCN on a synthetic stochastic block model.

## 4. PyTorch Geometric Baseline

```bash
python 03_pytorch_geometric/train_pyg_gcn.py --epochs 200
```

This reproduces the same learning problem with `Data` and `GCNConv`.

## 5. Build a Protein Graph

Download or place a PDB file, then run:

```bash
python 04_protein_graphs/pdb_to_graph.py data/pdb/1LYZ.pdb --chain-id A
```

If the file is not present yet, use the training or case-study scripts, which download required PDBs automatically.

## 6. Train the Protein Classifier

```bash
python 05_training/train_protein_classifier.py --epochs 80
```

Outputs are written to:

```text
artifacts/protein_classifier/
```

## 7. Reproduce the Minimal Relevant-Walk Demo

```bash
python 07_gnn_lrp/reproduce_demo.py
```

Outputs are written to:

```text
artifacts/gnn_lrp_demo/
```

## 8. Run Biological Case Studies

Lysozyme active-site explanation:

```bash
python 08_case_study/run_case_study.py --case-name lysozyme_active_site
```

Beta-lactamase mutation comparison:

```bash
python 08_case_study/run_case_study.py --case-name beta_lactamase_e104k
```

Outputs are written to:

```text
artifacts/case_studies/
```

## 9. Regenerate Case-Study Plots

```bash
python 08_case_study/visualize_case_study.py --case-json artifacts/case_studies/lysozyme_active_site/explanations.json --pdb-id 1LYZ --chain-id A
```

## 10. Regenerate Teaching GIFs

Render the full Manim storyboard and copy stable asset names into `assets/gifs/`:

```bash
python scripts/publish_manim.py --quality l
```

Useful options:

- `python scripts/publish_manim.py --scene ProteinBackboneScene --quality l`
- `python scripts/publish_manim.py --skip-render`

If GIF rendering fails, verify that `manim` and `ffmpeg` are available in the active environment.

## Expected Directory Outputs

- `data/pdb/`: downloaded structures
- `artifacts/protein_classifier/`: checkpoint and metrics
- `artifacts/gnn_lrp_demo/`: toy relevant-walk outputs
- `artifacts/case_studies/`: explanation JSON, plots, and PyMOL scripts
- `assets/gifs/`: stable teaching GIF exports such as `ProteinBackboneScene.gif`
