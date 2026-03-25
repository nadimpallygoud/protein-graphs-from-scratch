# Biological Case Studies

This module turns the general pipeline into concrete biological analyses.

## Case Study 1: Lysozyme Residue Importance

- protein: hen egg-white lysozyme (`1LYZ`, chain `A`)
- biological context: a classic enzyme with well-known catalytic residues
- expected important residues: `Glu35` and `Asp52`

The goal is to check whether the classifier and explanation pipeline focus on biologically meaningful parts of the structure.

## Case Study 2: Beta-Lactamase Mutation Comparison

- protein: TEM-1 beta-lactamase (`1BTL`, chain `A`)
- mutation: `E104K`
- biological context: beta-lactamase mutations can alter substrate and inhibitor behavior

This repository treats the mutation comparison as an **in silico feature mutation** on a real structure:

- the contact graph is kept fixed
- the residue identity and physicochemical features are changed at the mutation site
- prediction and relevance are compared between wild type and mutant

This is useful for mechanism-oriented hypothesis generation, even when a paired mutant structure is not available.

## Outputs

Running the case-study scripts produces:

- explanation JSON files
- graph plots with node and edge importance
- residue bar plots
- top relevant walks
- a PyMOL coloring script for 3D inspection

