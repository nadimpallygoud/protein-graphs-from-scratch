# Graph Basics for GNNs

This module explains the matrix language behind graph neural networks. The point is not to treat graphs as mysterious objects, but to show that a large fraction of GNN machinery can be expressed with a few linear-algebra building blocks.

## Intuition First

A graph is a collection of entities and relations:

- nodes are the entities
- edges are the relations

For proteins, nodes can be residues and edges can be spatial contacts. For social networks, nodes can be people and edges can be interactions.

The important idea is that once a graph is written as matrices, neighborhood aggregation becomes standard matrix multiplication.

## Adjacency Matrix

For a graph with `n` nodes, the adjacency matrix `A in R^{n x n}` stores connectivity:

```math
A_{ij} =
\begin{cases}
1, & \text{if there is an edge from } i \text{ to } j \\
0, & \text{otherwise}
\end{cases}
```

In an undirected graph, `A` is symmetric.

Why it matters:

- row `i` tells you which nodes send information to node `i`
- column `j` tells you where node `j` can send information
- multiplying `A @ X` sums neighbor features for each node

## Degree Matrix

The degree of a node is how many edges touch it. The degree matrix is diagonal:

```math
D_{ii} = \sum_j A_{ij}
```

Intuition:

- high-degree nodes have many neighbors
- if you do not normalize by degree, these nodes can dominate aggregation

## Graph Laplacian

The combinatorial Laplacian is:

```math
L = D - A
```

Interpretation:

- `A` spreads information across edges
- `D` keeps track of local mass at each node
- `L` measures how different a node is from its neighborhood

For a node signal `x`, the quadratic form:

```math
x^T L x
```

is small when connected nodes have similar values. This is why the Laplacian captures smoothness on a graph.

## Normalized Adjacency

Raw adjacency is usually not enough for learning because nodes with many neighbors accumulate larger values just because of degree. GNNs often use:

```math
\tilde{A} = A + I
```

```math
\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}
```

```math
\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}
```

Why this normalization works:

- adding `I` gives each node access to its own features
- left and right degree scaling makes aggregation numerically balanced
- the operator becomes a degree-aware averaging rule rather than a raw sum

## What the Code Implements

`graph_basics.py` contains:

- adjacency matrix construction
- degree matrix computation
- combinatorial and normalized Laplacians
- normalized adjacency used later by GCNs
- helper operations such as neighbor lookup and k-hop reachability

Run the demo script to see all matrices on a tiny hand-built graph.

