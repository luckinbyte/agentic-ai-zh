# System Architecture & Infrastructure at Scale

> 本章待翻译(原文页码:p199–p221)。


## The 4-Model Memory Challenge


## Parallelism Strategies in Detail


### Data Parallelism (DP) and Distributed Data Parallelism (DDP)


### Tensor Parallelism (TP)


### Sequence Parallelism (SP)


### Pipeline Parallelism (PP)


### Fully Sharded Data Parallelism (FSDP / ZeRO-3)


### 3D Parallelism: Combining Strategies


## The Generation Bottleneck: Quantitative Analysis


## Decoupled Architecture: Production Design


## Weight Synchronization Strategies


## Memory Optimization Techniques


### Flash Attention’s Impact on RLHF


## Fault Tolerance at Scale


## End-to-End Latency Breakdown


## Monitoring and Observability


## Network Topology and Communication Patterns


### Intra-Node: NVLink and NVSwitch


### Inter-Node: InfiniBand and RoCE


### Communication Primitives and Their Costs


### Network Topology Design


## Training Throughput and Model FLOPs Utilization


### Measuring Training Efficiency: MFU


### Compute-Optimal Batch Sizing


### Profiling and Bottleneck Diagnosis


## Cost Analysis and Cloud Deployment


### Hardware Cost Comparison


### RLHF Training Cost Estimation


### Cost Optimization Strategies


## Distributed Checkpointing


### Checkpointing Strategies


### Production Checkpointing with torch.distributed.checkpoint


## Hardware Selection Guide


## Optimizer Configuration for RL Training


### Why RL Requires Different Optimizer Settings


### Recommended Hyperparameters by RL Method


### Beta-2 = 0.95 for RL: Faster Adaptation


### Mixed Precision for RL: FP32 Master Weights Are Critical


### Gradient Clipping is Critical for RL


### Diagnosing RL Training Instability


### HuggingFace TRL Configuration for RL


### MoE Considerations for RL Training

