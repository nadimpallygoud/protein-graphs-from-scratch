# Suggested Extensions

## Modeling

- replace the dense GCN classifier with a sparse PyG graph-classification model for larger datasets
- compare GCN with GraphSAGE, GAT, and GIN
- add residual connections and normalization layers to study over-smoothing
- move from residue-level graphs to atom-level or surface-level graphs

## Protein Features

- add DSSP-derived secondary structure and solvent accessibility
- add pretrained protein language model embeddings per residue
- include edge features such as distance bins, hydrogen-bond indicators, or contact types
- represent chains, ligands, and cofactors explicitly

## Tasks

- mutation effect prediction on curated benchmarks
- residue importance prediction using catalytic-site annotations
- protein function or fold classification on larger structural datasets
- binding-site and interface prediction

## Explainability

- compare GNN-LRP with GNNExplainer, Integrated Gradients, and occlusion maps
- quantify explanation faithfulness with perturbation tests
- validate explanation overlap against known catalytic or binding residues
- extend relevant-walk extraction to longer paths or motif templates

## Biological Studies

- compare wild type and mutant relevance maps for disease variants
- analyze active-site conservation across homologs
- inspect allosteric pathways via high-scoring relevant walks
- export residue scores to PyMOL, ChimeraX, or nglview workflows

## Engineering

- add unit tests for graph construction and attribution conservation
- introduce Hydra or Pydantic configs for experiment management
- store processed graphs in an on-disk cache for faster iteration
- add CI to run syntax checks, lightweight training tests, and notebook smoke tests

