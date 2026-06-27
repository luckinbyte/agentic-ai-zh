# Systems Foundations for LLMs

> 本章待翻译(原文页码:p105–p118)。


## GPU Architecture – From Silicon to LLM Training


### Why GPUs for Deep Learning?


### NVIDIA GPU Microarchitecture Generations


### Common GPUs for LLM Training and Inference


### GPU Internal Architecture – The Streaming Multiprocessor (SM)


### GPU Chip Scaling Across Generations


### GPU Memory Hierarchy and Bandwidth


### Arithmetic Intensity and the Roofline Model


### Attention is Memory-Bound; FFN is Compute-Bound


### Tensor Cores


### Communication Architecture – NVLink, InfiniBand, and PCIe


## vLLM – PagedAttention and High-Throughput Inference


### The KV Cache Fragmentation Problem


### PagedAttention – Virtual Memory for KV Caches


### Benefits of PagedAttention


### Continuous Batching


### Speculative Decoding in vLLM


### Concrete Memory Savings – 70B Model at Scale


### vLLM: End-to-End System


### Architecture Overview


### Core Components


### Request Lifecycle (End-to-End Flow)


### Prefix Caching (Automatic Prompt Caching)


### Guided (Constrained) Decoding in vLLM

