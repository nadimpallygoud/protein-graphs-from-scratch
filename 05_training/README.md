# Training a Protein Graph Classifier

This module trains the main biological model used by the repository.

## Chosen Task

The primary supervised task is:

**protein classification from residue graphs**

This choice is deliberate:

- the target is graph-level, which matches protein-level labels naturally
- graph-level predictions are a clean setting for GNN-LRP
- residue relevance can still be extracted afterward
- mutation analysis becomes a comparison between two graphs under the same classifier

## Model

The baseline classifier is a dense GCN graph classifier:

1. apply graph convolutions to residue features
2. obtain residue embeddings
3. average them into a graph embedding
4. classify the protein from the pooled representation

The dense formulation is not the most scalable possible choice, but it makes the explanation rules transparent. That tradeoff is useful in an educational repository centered on interpretability.

## Dataset

The included demo metadata uses real PDB structures:

- enzymes such as lysozyme, beta-lactamase, and adenylate kinase
- non-enzyme proteins such as ubiquitin, hemoglobin, and streptavidin

The dataset is intentionally small. It is a tutorial dataset for wiring the end-to-end pipeline, not a benchmark claim.

## Output

Training produces:

- a saved model checkpoint
- JSON metrics
- a list of graph names used for training and testing

