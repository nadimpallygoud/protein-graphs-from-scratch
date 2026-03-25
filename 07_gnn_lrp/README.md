# GNN-LRP

This module implements a pedagogical version of GNN-LRP for dense GCN graph classifiers.

## What Is LRP

Layer-wise relevance propagation does not ask:

> what would happen if I changed the input slightly?

Instead, it asks:

> how should the prediction that was actually made be redistributed back to the input components that supported it?

This is a contribution-tracing view of explanation rather than a local sensitivity view.

## Conservation Principle

The key idea is relevance conservation:

```math
\sum_i R_i^{(l)} = \sum_j R_j^{(l+1)}
```

The total amount of relevance should stay consistent as it moves backward through the model. In practice, small stabilizers are used to avoid division by zero, but the conceptual goal is that the prediction evidence is reassigned, not created from nothing.

## How GNN-LRP Extends LRP

GNNs are harder than MLPs because:

- information is passed across edges
- the same node can contribute to multiple downstream neighborhoods
- important evidence can live in paths and subgraphs, not isolated nodes

GNN-LRP adapts relevance propagation to this setting by following the graph message-passing structure. The result is not only node scores, but also edge and walk importance.

## Walk-Based Explanations

A graph prediction can depend on a chain of interactions:

- residue A supports residue B
- residue B supports residue C
- the combined local motif supports the final prediction

Relevant walks make this explicit by ranking high-evidence paths through the graph.

## Scope of This Implementation

This repository follows the spirit of the public GNN-LRP demos and the Schnake et al. paper, but simplifies the bookkeeping to a dense GCN classifier so the propagation rules remain easy to inspect and adapt. It is therefore:

- faithful to the key intuition of relevance conservation and relevant walks
- suitable for educational and small-scale biological experiments
- intentionally simpler than the most general sparse-graph research implementations

