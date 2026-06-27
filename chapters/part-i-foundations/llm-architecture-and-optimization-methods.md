# LLM Architecture and Optimization Methods

> 本章待翻译(原文页码:p35–p104)。


## How LLMs Work: An Intuitive Overview


## Tokenization


### Why Not Characters or Words?


### Byte-Pair Encoding (BPE)


### Other Tokenization Methods


### Tokenization Best Practices


### Tokenization in Practice: HuggingFace Example


### Special Tokens and Structured Prompts


## The Transformer Architecture


### High-Level Structure


### The Original Encoder-Decoder Transformer


### Decoder-Only vs Encoder-Decoder


### Embeddings: From Discrete Tokens to Continuous Space


### Self-Attention Mechanism


### Multi-Head Attention


### Positional Encodings


### Feed-Forward Network (MLP)


### Layer Normalization


### Model Size Reference


### Attention Pathologies


### Visualizing Attention for Explainability


## Prediction Heads: What Transformers Output


### Language Modeling Head (Pretraining)


### Conditional Generation Head (SFT / Instruction Following)


### Value Head (Regression for RL)


### Head Selection Summary


### HuggingFace Implementation


## Optimization Theory for LLM Training


### Gradient Descent: The Foundation


### Why Vanilla SGD Fails for LLMs


### Adam – Adaptive Moment Estimation


### AdamW – Decoupled Weight Decay


### Learning Rate – The Most Important Hyperparameter


### Learning Rate Warmup


### Learning Rate Schedules


### Gradient Clipping


### Mixed Precision Training


### Practical Optimizer Settings by Training Phase


## Flash Attention – Algorithm and Hardware Awareness


### The Standard Attention Memory Problem


### The Flash Attention Key Insight – Tiling and Online Softmax


### The Flash Attention Algorithm


### Flash Attention 2 – Better Parallelism


### Flash Attention 3 – Hopper Architecture


### Flash Attention 4 – Blackwell Architecture


## Pretraining: Best Practices


### Training Objective


### Data Pipeline


### Scaling Laws


### Key Hyperparameters


### Common Failure Modes


## Supervised Fine-Tuning (SFT)


### SFT Objective


### Data Quality: The LIMA Principle


### Training Configuration


### Efficient Training Solutions


### Best Practices


## LoRA and Parameter-Efficient Fine-Tuning


### The LoRA Insight


### LoRA Hyperparameters


### LoRA Variants


### Other PEFT Approaches


## Mixture of Experts (MoE)


### Architecture


### Load Balancing


### Noisy Top-K Gating: Making Discrete Routing Trainable


### Notable MoE Models


## Diversity in LLM Training


### Sampling Diversity


### Training Data Diversity


### Diversity-Promoting Methods


## Text Generation: Decoding Methods


### Greedy Decoding


### Beam Search


### Diverse Beam Search


### Top-k Sampling


### Top-p (Nucleus) Sampling


### Min-p Sampling


### Temperature Scaling


### Contrastive Decoding


### Repetition Penalties


### Practical Comparison


### Constrained Decoding (Structured Generation)


## Prompt Engineering


### In-Context Learning (ICL)


### Zero-Shot Prompting


### Few-Shot Prompting


### Instruction-Following Prompts


### Structured Output Prompts (JSON/XML)


### Chain-of-Thought (CoT) Prompting


### Advanced Prompting Techniques


### Best Practices: Crafting Effective Prompts


## Model Compression Methods


### Quantization


### Pruning


### Knowledge Distillation


## Speculative Decoding Methods


### Core Principle


### Methods Comparison


### Medusa: Multi-Head Speculative Decoding


### Eagle: Feature-Level Drafting


### N-gram Speculative Decoding


### Integration with vLLM


## Hallucination Detection


### Types of Hallucination


### Detection Methods (Model-Level)


## LLM Safety and Responsible AI


### Threat Taxonomy


### Safety Training Pipeline


### Key Safety Mechanisms


### The Helpfulness–Safety Tradeoff


### Evaluation

