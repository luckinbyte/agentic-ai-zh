# PPO — Proximal Policy Optimization

> 本章待翻译(原文页码:p136–p144)。


## Motivation and History


## The Clipped Objective


## Full PPO Loss


## Derivation of the PPO Gradient and Update Rule


### Step 1: The RL Objective


### Step 2: Policy Gradient Theorem


### Step 3: Importance Sampling for Off-Policy Data


### Step 4: The Problem with Unconstrained Surrogates


### Step 5: PPO’s Clipped Surrogate (First-Order Approximation)


### Step 6: The Complete PPO Update Rule


## Rollout Buffer and Rollouts


### What is a Rollout?


### The Rollout Buffer


### The Rollout Buffer Lifecycle


## PPO for RLHF: The Full Loop


## Detailed Mechanics: Logits and Policy Updates


### Phase 1: Rollout (Data Collection)


### Phase 2: Optimization Loop (Mini-Batch Updates)


### From Logits to Probability Ratio


### The PPO Weight Lifecycle


### Continuous Action Spaces Extension


## TRL Implementation


## Critical Hyperparameters

