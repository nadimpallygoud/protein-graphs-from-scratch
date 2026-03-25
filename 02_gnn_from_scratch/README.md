# GNN Fundamentals From Scratch

This module builds the graph convolutional network idea directly from the propagation rule:

```math
H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})
```

## Message Passing

At layer `l`, each node has a feature vector stored in row `i` of `H^{(l)}`.

The update does three things:

1. `H^{(l)} W^{(l)}` mixes feature channels
2. `\hat{A} (...)` aggregates transformed neighbor information
3. `sigma` adds nonlinearity

Intuitively:

- a node first decides what information it wants to send
- the graph structure routes that information to neighbors
- repeated layers let each node build a representation of its local subgraph

## Why Add Self-Loops

Without self-loops, a node could lose direct access to its own signal after one layer. Adding the identity matrix means each node keeps a channel for its own information while also listening to neighbors.

## Why Normalize

If node `i` has many neighbors, a raw sum can grow just because the node is highly connected. Normalization makes aggregation behave more like a degree-aware averaging operation. This stabilizes learning and makes the update less sensitive to graph density.

## Why It Works

Graph convolution works because many graph tasks depend on local context:

- community membership in social graphs
- chemical context in molecules
- residue neighborhoods in proteins

Neighboring nodes often provide the evidence needed to classify a node or a whole graph. A GCN turns this intuition into a differentiable operator.

## Over-Smoothing

If many graph-convolution layers are stacked, node embeddings can become too similar:

- information diffuses repeatedly
- distinct local signals are averaged away
- classification becomes harder

This is called over-smoothing. It is one reason educational repositories should begin with shallow GCNs before moving to deeper architectures and residual designs.

## What the Code Implements

`models.py` contains:

- adjacency normalization in Torch
- a dense GCN layer
- a two-layer node classifier
- training helpers for the synthetic graph

`train_scratch_gcn.py` trains the model on a small stochastic block model so the mechanics stay transparent.

