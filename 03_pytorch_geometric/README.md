# PyTorch Geometric Reimplementation

This module re-expresses the same GCN ideas using PyTorch Geometric.

## Why PyG Matters

Once the message-passing idea is clear, PyTorch Geometric becomes useful because it handles:

- sparse graph storage
- batching
- standard layers such as `GCNConv`
- common training patterns

The goal is not to replace understanding with library calls. The goal is to show that PyG is implementing the same ideas in a research-friendly form.

## Core Concepts

### `Data`

`torch_geometric.data.Data` stores a graph as tensors:

- `x`: node features
- `edge_index`: graph connectivity in coordinate form
- `y`: labels

### `GCNConv`

`GCNConv` performs the normalized neighborhood aggregation associated with the Kipf and Welling GCN. It hides sparse bookkeeping, but conceptually it still computes a normalized message-passing update.

### Training Loop

The training loop is still standard deep learning:

1. forward pass
2. compute loss
3. backpropagate
4. update weights
5. evaluate on held-out nodes or graphs

The difference is that the model now consumes sparse graph objects rather than dense adjacency matrices.

