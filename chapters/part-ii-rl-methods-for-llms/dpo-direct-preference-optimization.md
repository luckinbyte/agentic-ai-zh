# DPO — Direct Preference Optimization

> 本章待翻译(原文页码:p145–p157)。


## Motivation


## Mathematical Derivation


## Gradient Analysis


## TRL Implementation


## How DPO Works: Full Mechanics


### Sequence-Level Log-Probabilities


### The DPO Loss Decomposed


### Forward Pass: Step by Step


### Token-Level Gradient Analysis


### Per-Token vs. Sequence-Level: Length Normalization


### Label Masking: What Gets Gradients


### Pseudocode: DPO Training Step


### Common Pitfalls


## DPO Variants and When Each Fails


## Selection Guide


## DPO Batch Size Configuration and Scaling


### Global Batch Size Target


### Mathematical Decomposition


### Distributed Scaling Configurations


### VRAM Optimization: Pre-computing Reference Log-Probabilities


## DPO Extensions and Variants


### f-DPO – Generalised f-Divergence DPO


### Robust DPO


### TR-DPO – Trust Region DPO


### EXO – Exact Optimisation


### NCA – Noise Contrastive Alignment


### SLiC-HF – Sequence Likelihood Calibration


### Iterative RPO – Reasoning Preference Optimisation


### SimPO – Simple Preference Optimisation

