# Explainability Before GNN-LRP

This module introduces simpler attribution methods before moving to relevance propagation.

## Why Start With Baselines

Good explanation work starts by comparing methods, not by assuming one technique is always right.

The baseline explainers here answer different questions:

- gradients ask how sensitive the prediction is to small changes
- node occlusion asks how much the score drops if a residue is hidden
- edge occlusion asks how much the score drops if a contact is removed

These are useful reference points because they make the limitations of naive saliency visible.

## Interpretability Challenges in Graphs

Graph explanations are difficult because:

- signal is mixed repeatedly through message passing
- a node can matter because of its neighborhood, not just its own features
- structure and features interact
- local derivatives can be noisy or unstable

In proteins, these problems are amplified because biological function often depends on cooperative residue neighborhoods.

## Why GNN-LRP Comes Next

Gradient and masking methods are good sanity checks, but they do not enforce relevance conservation and they do not naturally highlight higher-order interaction paths. That gap motivates the GNN-LRP module.

