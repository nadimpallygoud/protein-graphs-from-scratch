 # GNNs for Protein Graphs + Explainability with GNN-LRP

An end-to-end educational and research-grade repository for learning graph neural networks from first principles, applying them to protein structure graphs, and interpreting predictions with GNN-LRP.

This project is designed to move in a strict progression:

1. graph theory basics
2. GNN fundamentals from scratch
3. PyTorch Geometric implementations
4. protein graph construction from PDB structures
5. model training on a biologically meaningful task
6. explainability with GNN-LRP
7. visualization of important residues and interactions
8. biological case studies, including mutation analysis

The intended result is a repository that is useful in three ways at once:

- as a tutorial for readers starting from zero
- as a practical codebase for protein-graph experiments
- as a portfolio-quality or publication-ready foundation for explainable structural bioinformatics

## Project Status

The repository now includes the full first implementation pass:

- mathematical walkthroughs for graph basics, GCNs, protein graphs, explainability, and GNN-LRP
- modular Python code from Numpy graph operations to dense GCN protein classification
- notebook-based learning material
- residue-graph construction from real PDB structures
- baseline explainability and GNN-LRP relevant-walk attribution
- two biological case studies with saved artifacts
- lightweight tests and CI scaffolding for regression protection

The demo dataset and case studies are intentionally small and educational. The repository is now usable end to end, but it remains designed to be extended with larger curated datasets and stronger experimental protocols.

## Quickstart

For a complete run sequence, see [RUNNING.md](RUNNING.md). The shortest end-to-end path is:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python 05_training/train_protein_classifier.py --epochs 80
python 08_case_study/run_case_study.py --case-name lysozyme_active_site
python 08_case_study/run_case_study.py --case-name beta_lactamase_e104k
```

## Implemented Outputs

The repository currently ships with:

- graph basics in [01_graph_basics/](01_graph_basics/)
- a scratch dense GCN in [02_gnn_from_scratch/](02_gnn_from_scratch/)
- a PyTorch Geometric baseline in [03_pytorch_geometric/](03_pytorch_geometric/)
- residue-graph construction utilities in [04_protein_graphs/](04_protein_graphs/)
- protein graph classification in [05_training/](05_training/)
- gradient and occlusion baselines in [06_explainability/](06_explainability/)
- GNN-LRP and relevant walks in [07_gnn_lrp/](07_gnn_lrp/)
- biological case studies in [08_case_study/](08_case_study/)
- notebook tutorials in [notebooks/](notebooks/)
- test coverage in `tests/`

Generated figures, explanation JSON files, and checkpoints are written to `artifacts/` during execution and are ignored by Git.

## Repository Quality

This repository now includes:

- a `.gitignore` for generated data and artifacts
- a small `pytest` suite for core math, graph construction, and GNN-LRP regression checks
- a GitHub Actions workflow for compile and test checks
- an MIT [LICENSE](LICENSE)

## Why This Repository Exists

Proteins are not naturally sequences alone. Their biological function depends on spatial proximity, residue-residue interactions, geometry, chemistry, and context. Graphs are a natural representation for this setting:

- nodes can represent residues or atoms
- edges can represent spatial contacts or biochemical relations
- node features can encode amino-acid identity and physicochemical properties
- graph structure preserves long-range interactions that sequence-only models may obscure

Graph neural networks are therefore a strong fit for protein modeling. However, predictive accuracy is not enough in biology. A useful model must also answer:

- which residues matter?
- which interactions support the prediction?
- does the explanation align with known catalytic, binding, or mutation-sensitive regions?

This is where GNN-LRP becomes important. Unlike raw gradients, GNN-LRP is built to redistribute prediction evidence through the network in a way that respects relevance conservation and exposes higher-order interaction patterns as relevant walks.

## Central Question

Given a protein structure, can we:

1. convert it into a residue graph,
2. train a GNN to solve a biologically meaningful prediction task,
3. explain the prediction in terms of residues, contacts, and structural subgraphs,
4. compare explanations across wild-type and mutated proteins?

This repository is organized to answer that question from first principles.

## Audience

This repository is written for:

- students learning graph neural networks for the first time
- bioinformatics researchers who want a clear protein-graph pipeline
- XAI practitioners interested in adapting LRP to relational models
- engineers who want clean, modular, reproducible code instead of a one-off notebook

No prior expertise in GNN-LRP is assumed. Basic Python, linear algebra, and introductory deep learning are helpful, but the repository is designed to explain concepts carefully and incrementally.

## What You Will Learn

By the end of the full repository, the reader should be able to:

- understand adjacency matrices, degree matrices, and graph Laplacians
- implement a simple graph convolutional layer from scratch
- understand message passing and adjacency normalization
- use `torch_geometric.data.Data` and `GCNConv`
- parse PDB files and build residue-level protein graphs
- engineer residue features from amino-acid identity and physicochemical descriptors
- train a GNN on a protein-related task
- distinguish gradient-based explanations from relevance propagation
- understand the conservation principle in LRP
- implement walk-based explanations for GNNs
- visualize important residues on both the graph and the protein structure
- compare explanation maps between wild-type and mutant structures

## Repository Roadmap

The repository is intentionally structured as a teaching pipeline. Each folder corresponds to a conceptual jump.

| Path | Purpose |
| --- | --- |
| `01_graph_basics/` | Numpy-based graph operations: adjacency, degree, Laplacian, graph intuition |
| `02_gnn_from_scratch/` | Minimal GCN implementation from first principles |
| `03_pytorch_geometric/` | Practical GNN workflow using PyTorch Geometric |
| `04_protein_graphs/` | PDB parsing, residue graph construction, feature extraction |
| `05_training/` | Training, evaluation, metrics, and experiment configuration |
| `06_explainability/` | Introductory explainability baselines for GNNs |
| `07_gnn_lrp/` | GNN-LRP implementation, relevance propagation, relevant walks |
| `08_case_study/` | Biological examples: active-site importance and mutation analysis |
| `notebooks/` | Notebook-based teaching material and walkthroughs |
| `utils/` | Shared utilities, constants, plotting, and helper functions |

## Folder Structure

```text
gnn-protein-lrp/
|
|-- README.md
|-- requirements.txt
|-- 01_graph_basics/
|-- 02_gnn_from_scratch/
|-- 03_pytorch_geometric/
|-- 04_protein_graphs/
|-- 05_training/
|-- 06_explainability/
|-- 07_gnn_lrp/
|-- 08_case_study/
|-- notebooks/
`-- utils/
```

## Technical Scope

### 1. Graph Theory Basics

The project begins with concrete matrix objects:

- adjacency matrix `A`
- degree matrix `D`
- combinatorial Laplacian `L = D - A`
- normalized graph operators

These foundations matter because many GNN layers are best understood as learnable, normalized neighborhood aggregation operators.

### 2. GNN Fundamentals From Scratch

The repository implements a simple graph convolution layer based on the standard normalized propagation rule:

```math
H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})
```

where:

- `H^{(l)}` is the node-feature matrix at layer `l`
- `W^{(l)}` is a learnable weight matrix
- `sigma` is a nonlinearity
- `\hat{A}` is the normalized adjacency with self-loops

This section explains:

- what message passing means operationally
- why self-loops are added
- why normalization stabilizes learning
- how repeated neighborhood mixing creates useful structural embeddings

### 3. PyTorch Geometric

After the scratch implementation, the same ideas are reimplemented using PyTorch Geometric:

- `Data` objects for graph representation
- `GCNConv` for graph convolutions
- batching, loaders, and training loops

The goal is not to hide the math behind the library, but to connect theory to a practical framework used in research.

### 4. Protein Graph Construction

![Protein Topology](assets/gifs/ProteinBackboneScene.gif)


Protein structures are converted into residue graphs using PDB files.

Default representation:

- node: one amino-acid residue
- edge: residue-residue contact under a distance threshold, such as 8 angstrom, based on `CA` coordinates or representative heavy atoms
- node features:
  - amino-acid identity as one-hot encoding
  - physicochemical descriptors such as polarity, charge, hydrophobicity, aromaticity, and size
  - optional structural signals such as solvent accessibility, secondary structure, or local geometry

This abstraction is biologically meaningful because residues far apart in sequence can be near in 3D space and jointly determine function.

### 5. Predictive Task

The primary implemented supervised task in this repository is:

**protein classification from residue graphs**

This is the most stable educational starting point because it:

- supports graph-level prediction cleanly
- aligns well with GNN-LRP graph explanations
- allows biologically interpretable residue-level attributions
- can be extended naturally to mutation perturbation analysis

Natural extension tasks:

- residue importance prediction
- mutation effect comparison between wild-type and mutant graphs

### 6. Explainability

Before GNN-LRP, the repository introduces simpler explanation families:

- saliency and gradient-style scores
- feature masking intuition
- subgraph importance intuition

This makes the limitations of naive gradients visible before moving to relevance propagation.

### 7. GNN-LRP

The core explainability component of the repository is GNN-LRP:

- relevance is propagated backward through the trained GNN
- the explanation respects a conservation principle
- importance is assigned to higher-order graph structures, not only isolated nodes
- explanations can be expressed as relevant walks through the graph

This is especially attractive in proteins, where function often depends on coordinated residue neighborhoods rather than a single residue in isolation.

### 8. Visualization

The repository visualizes explanations in multiple complementary ways:

- 2D graph plots with important nodes and edges highlighted
- residue-level heatmaps
- structure-aware visual overlays for protein coordinates
- side-by-side wild-type versus mutant importance comparisons

The Manim storyboard is organized as a teaching sequence rather than a loose GIF dump:

| Scene | Key concept | Why it matters |
| --- | --- | --- |
| `GraphBasicsScene` | nodes, edges, adjacency rows | grounds later message passing in explicit graph bookkeeping |
| `GCNMessagePassingScene` | self-loop plus normalized neighbor aggregation | explains what one GCN layer actually computes |
| `FeatureVectorUpdateScene` | explicit feature-vector averaging | shows that embeddings are numeric combinations, not abstract color changes |
| `PyGEdgeIndexScene` | sparse `edge_index` storage | connects the math view to the PyTorch Geometric implementation |
| `ProteinBackboneScene` | sequence distance vs 3D contact | motivates why protein graphs need structure-aware edges |
| `ProteinDistanceGraphScene` | threshold-based contact graph construction | shows how residue coordinates become graph connectivity |
| `GraphPoolingScene` | graph-level pooling | explains how node embeddings become one protein prediction |
| `SaliencyVsSubgraphScene` | pointwise saliency vs connected evidence | motivates the move from gradients to structural explanations |
| `LRPNumericConservationScene` | relevance conservation | explains the accounting principle behind LRP |
| `GNNLRPRelevantWalkScene` | relevant walks | shows why GNN explanations can be path- or subgraph-level |
| `WildtypeMutantDeltaScene` | explanation shifts after mutation | connects the visualization track to the biological case studies |

To regenerate the full set of teaching GIFs:

```bash
python scripts/publish_manim.py --quality l
```

## Mathematical Preview

### Basic Graph Objects

For a graph with `n` nodes:

```math
A_{ij} =
\begin{cases}
1, & \text{if nodes } i \text{ and } j \text{ are connected} \\
0, & \text{otherwise}
\end{cases}
```

```math
D_{ii} = \sum_j A_{ij}
```

```math
L = D - A
```

The Laplacian measures how each node differs from its neighborhood and is central to spectral graph methods.

### Normalized Propagation

![Message Passing](assets/gifs/FeatureVectorUpdateScene.gif)


For GCNs, we usually add self-loops and normalize:

```math
\tilde{A} = A + I
```

```math
\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}
```

```math
\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
```

This produces the standard update:

```math
H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})
```

Interpretation:

- each node collects information from neighbors
- self-loops preserve the node's own signal
- symmetric normalization prevents high-degree nodes from dominating the aggregation

### Relevance Conservation

![GNN LRP](assets/gifs/LRPNumericConservationScene.gif)


Layer-wise relevance propagation redistributes a model output score backward through the network:

```math
\sum_i R_i^{(l)} = \sum_j R_j^{(l+1)} = f(x)
```

Informally, the prediction score is not treated as a derivative but as evidence that must be reassigned to lower-level components.

For GNNs, this becomes more subtle because:

- the graph itself is part of the computational structure
- information is repeatedly mixed across edges
- important evidence may emerge from a sequence of interactions, not a single local term

GNN-LRP addresses this by decomposing predictions into contributions associated with graph walks and higher-order interaction patterns.

## Why Proteins Are Naturally Graphs

Proteins are well modeled as graphs because:

- residues act as structured units with attributes
- non-local contacts are central to folding and function
- active sites often involve residue groups rather than isolated positions
- mutations can change local features and long-range interaction patterns

Graph representations capture what sequence-only or voxel-based abstractions can miss:

- contact topology
- structural context
- neighborhood chemistry
- paths linking distant but functionally coupled residues

## Why Explainability Matters in Protein GNNs

A protein model that predicts correctly but cannot justify itself is limited. In biological settings, explanations are needed to:

- validate whether the model focuses on catalytic or binding residues
- detect shortcuts or dataset artifacts
- generate mechanistic hypotheses
- prioritize mutations or wet-lab follow-up

This repository emphasizes explanation quality, not only predictive accuracy.

## Gradients vs LRP

This distinction is central to the project.

### Gradient-Based Explanations

Gradient explanations ask:

> how much would the output change if the input changed infinitesimally?

They are useful, but can be noisy or unstable on graph inputs, and they do not explicitly enforce conservation of evidence.

### Layer-Wise Relevance Propagation

LRP asks:

> how much did each component contribute to the prediction that was actually made?

This is an attribution perspective rather than a local sensitivity perspective.

In practice, LRP is often better aligned with the goal of tracing predictive evidence back to interpretable input components.

### Why GNN-LRP Is Special

In graphs, evidence can arise from combinations of nodes and edges that only become meaningful together. GNN-LRP explicitly supports this by identifying relevant walks and interaction patterns instead of reducing the explanation to isolated pointwise scores.

## Over-Smoothing and Other Modeling Challenges

The theory sections of this repository also explain common GNN issues:

- over-smoothing: repeated aggregation can make node embeddings too similar
- over-squashing: too much information is compressed into limited embeddings
- topology bias: some graph constructions emphasize geometry while hiding chemistry
- explanation ambiguity: important residues may appear in multiple interacting walks

These issues matter directly for proteins, where both local chemistry and long-range structure can be decisive.

## Biological Case Studies

The repository includes small biological examples with structure-driven explanations.

### Case Study 1: Functionally Important Residues

A known enzyme or binding protein is processed as a residue graph, classified by the model, and explained with GNN-LRP. The explanation is then checked against known functionally important residues such as catalytic, binding, or structurally stabilizing positions.

### Case Study 2: Mutation Comparison

A wild-type and mutated protein pair is compared by:

- building matched graphs
- running the trained model on both forms
- computing GNN-LRP scores
- visualizing how relevance shifts around the mutation site and its structural neighborhood

This case study is intended to demonstrate how explainability can support mechanistic reasoning, not just predictive reporting.

## Reference Papers

The repository is grounded in a small set of foundational papers that are explained in plain language and connected to code.

1. **Kipf and Welling (2017), "Semi-Supervised Classification with Graph Convolutional Networks"**  
   Core idea: approximate spectral graph convolutions with a simple normalized message-passing rule that is efficient and widely usable.  
   Link: https://arxiv.org/abs/1609.02907

2. **Ying et al. (2019), "GNNExplainer: Generating Explanations for Graph Neural Networks"**  
   Core idea: explain a trained GNN by learning a compact subgraph and feature mask that preserve the prediction.  
   Link: https://papers.nips.cc/paper/9123-gnnexplainer-generating-explanations-for-graph-neural-networks

3. **Schnake et al. (2022), "Higher-Order Explanations of Graph Neural Networks via Relevant Walks"**  
   Core idea: extend relevance propagation to GNNs so predictions can be decomposed into higher-order graph contributions expressed as relevant walks.  
   Link: https://doi.org/10.1109/TPAMI.2021.3115452

4. **Gainza et al. (2020), "Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning"**  
   Core idea: protein structure can be learned from geometric representations in a biologically meaningful way, reinforcing the broader case for structure-aware deep learning in proteins.  
   Link: https://doi.org/10.1038/s41592-019-0666-6

## Reference Implementations

Two public implementations are especially relevant as inspiration:

- original `demo_gnn_lrp` repository by Thomas Schnake and collaborators: https://git.tu-berlin.de/thomas_schnake/demo_gnn_lrp
- simplified public reimplementation: https://github.com/liwenke1/GNN-LRP

This repository does not copy those codebases directly. Instead, it:

- rewrite the ideas in a smaller and more educational form
- separate theory from engineering details
- adapt the methodology to protein graphs and biological interpretation

## Design Principles

The implementation follows a few strict principles:

- modular code with minimal hidden state
- comments that explain why a design choice exists
- reproducible splits and random seeds
- compact dependencies where possible
- explicit data preprocessing
- educational clarity without sacrificing correctness

## Installation

The repository uses a lightweight Python stack centered on Numpy, PyTorch, PyTorch Geometric, BioPython, Manim, and plotting utilities.

Typical setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Notes:

- `torch-geometric` installation can depend on your PyTorch and CUDA setup.
- the first pass of this repository is CPU-friendly; GPU support is optional
- BioPython is the default dependency for PDB parsing to keep the stack minimal
- Manim GIF rendering also expects a working local multimedia stack, including `ffmpeg`

## Repository Deliverables

The repository contains:

- mathematical walkthroughs from graph basics to GCNs
- clean Python modules for each stage
- notebook tutorials for self-study
- residue-graph builders from raw PDB files
- training scripts for a protein-related task
- GNN-LRP code for node, edge, and walk relevance
- figure-generation scripts for graph and structure visualizations
- biologically interpretable case studies

## What Makes This Repository Different

Many graph-learning repositories do one of two things:

- provide elegant theory but little biological grounding
- provide biological experiments but weak conceptual explanation

This project is designed to bridge that gap. The intent is to make every modeling choice legible:

- why residues are chosen as nodes
- why a distance threshold is used for edges
- why normalized message passing works
- why gradient saliency is not enough
- why relevant walks are a natural explanation unit for graph models


## Citation and Credit

If you use ideas from this repository in research or teaching, please cite the primary papers above and credit this repository appropriately.

## Summary

This repository is a structured path from:

- graph matrices
- to message passing
- to protein graphs
- to explainable structural learning

The long-term objective is not only to build a working protein GNN, but to make its predictions biologically inspectable through GNN-LRP.
#   p r o t e i n - g r a p h s - f r o m - s c r a t c h 
 
 
