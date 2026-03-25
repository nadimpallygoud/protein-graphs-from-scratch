# Protein Graph Construction

This module turns protein structures into residue graphs.

## Why Proteins as Graphs

Protein function depends on 3D structure. Residues that are distant in sequence can be adjacent in space, and these long-range contacts often drive catalysis, binding, allostery, and mutation sensitivity.

A residue graph captures this naturally:

- node = one residue
- edge = a residue-residue contact in 3D space
- node features = amino-acid identity plus chemistry

This representation is more biologically faithful than sequence-only locality when the task depends on spatial interaction patterns.

## Graph Construction Choices

### Nodes

Each node is a standard amino-acid residue. Non-standard residues, waters, and ligands are skipped in the default educational pipeline so the graph remains easy to interpret.

### Edges

An undirected edge is added when two residues are within a distance threshold in Cartesian space. The default uses representative residue coordinates and a threshold of 8.0 angstrom.

This creates a contact graph that approximates the structural neighborhood of each residue.

### Node Features

Each residue receives:

- a 20-dimensional one-hot encoding for amino-acid identity
- a 5-dimensional physicochemical descriptor vector

These features are intentionally simple. The reader can inspect, mutate, and extend them without needing a large feature-engineering stack.

## Biological Interpretation

This representation makes it possible to ask:

- which residue neighborhoods support a classification?
- do high-relevance residues overlap with catalytic residues?
- how does a mutation alter local evidence flow through the graph?

## What the Code Implements

- `fetch_pdb.py`: download PDB files from RCSB
- `pdb_to_graph.py`: parse structures and build residue graphs
- `ProteinGraph` in `utils/protein_graph.py`: shared container for downstream training and explanation

