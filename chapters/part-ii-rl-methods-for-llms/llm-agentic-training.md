# LLM Agentic Training

> 本章待翻译(原文页码:p222–p249)。


## Motivation: From Chatbots to Autonomous Agents


## Trajectory Buffers for LLM Agents


### Mathematical Structure of an LLM Agent Buffer


## Operational Paradigms


### A. Self-Correction and Thought Refinement


### B. Off-Policy Exploration


### C. Non-Parametric In-Context Learning (RAG over Experiences)


## Paradigm Comparison


## Major Techniques in Agentic RL


### STaR: Self-Taught Reasoner (Detailed)


### Reflexion: Verbal Reinforcement Learning (Detailed)


### ReAct: Reasoning + Acting (Detailed)


### LATS: Language Agent Tree Search (Detailed)


### AgentQ: DPO on Agent Trajectories (Detailed)


### Voyager: Lifelong Learning via Skill Libraries (Detailed)


### RLEF: RL from Execution Feedback (Detailed)


### OpenHands / SWE-Agent: GRPO for Software Engineering


## Use Case: Agentic RL for a Productivity Co-pilot


### Architecture Overview


### Formal MDP Definition for a Productivity Co-pilot


### Action Space Design


### State Representation


### Reward Design: Multi-Objective Signal


### Training Pipeline: End-to-End


### Simulation Environment Architecture


### Task Curriculum Design


### Safety and Guardrails


### Credit Assignment in Multi-App Workflows


### Scaling and Infrastructure


### Evaluation Framework


### Lessons from Production Deployments


### Complete Training Recipe


## Use Case: Building a Research Agent from Scratch


### Problem Definition


### MDP Formulation


### Action Space


### Architecture: Model and Infrastructure Choices


### Reward Design


### Training Pipeline


### Example Trajectory: Full MDP Trace


### Key Design Decisions and Tradeoffs


### Evaluation


### Lessons and Failure Modes


## State-of-the-Art RL for LLM Agents


### Dominant Baseline: GRPO for Agents


### PPO for Interactive Agents


### Fine-Grained Turn-Level Credit Assignment


### Alternative Paradigms


### Core Methodology Comparison

