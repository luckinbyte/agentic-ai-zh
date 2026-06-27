<!-- page 71 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Quality filtering: Perplexity-based classifier, heuristic filters (length, language ID, toxicity)
• Data mixing: Temperature-weighted sampling across domains; upweight code and math
for reasoning
1.7.3
Scaling Laws
Hoffmann et al. [85] showed that compute-optimal training requires balancing model size N and data
size D: Nopt ∝C0.50, Dopt ∝C0.50. A 70B model is compute-optimal at ∼1.4T tokens. In practice,
models are over-trained (more tokens than Chinchilla-optimal) because inference cost scales with
model size, not training tokens—smaller over-trained models are cheaper to deploy.
1.7.4
Key Hyperparameters
Table 1.10: Pretraining hyperparameters from published models.
Setting
Llama-3
405B
Llama-3 8B
Qwen-2.5 72B
Mistral 7B
Tokens
15T
15T
18T
8T
Batch size (tokens)
16M
4M
4M
4M
Peak LR
8e-5
3e-4
3e-4
3e-4
Schedule
WSD
WSD
Cosine
Cosine
Weight decay
0.1
0.1
0.1
0.1
Context length
8192
8192
4096→32K
8192
1.7.5
Common Failure Modes
Pretraining Pitfalls
• Loss spikes: Sudden loss increases from bad data batches or numerical instability. Llama-3
reports rolling back to checkpoints and skipping offending batches.
• Memorization: Model regurgitates training data verbatim. Fix: deduplicate aggressively;
monitor extraction attacks.
• Context length: Training on short sequences then deploying at long context fails. Use
continued pretraining on long documents + RoPE scaling.
1.8
Supervised Fine-Tuning (SFT)
SFT transforms a pretrained language model into an instruction-following assistant by training on
curated prompt–response pairs. This is the bridge between raw language modeling and RLHF.
1.8.1
SFT Objective
The loss is identical to CLM, but computed only on response tokens:
LSFT = −1
|y|
|y|
X
t=1
log Pθ(yt | xprompt, y<t)
Prompt tokens provide context but receive no gradient (labels set to −100).
1.8.2
Data Quality: The LIMA Principle
Zhou et al. [87] demonstrated that 1,000 carefully curated examples can match models trained on
50K+ noisy examples. Key requirements:
71


<!-- page 72 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Diversity: Cover QA, summarization, code, math, creative writing, multi-turn dialogue
• Correctness: Every response must be factually accurate and well-formatted
• Length balance: Mix short (1-sentence) and long (multi-paragraph) responses
• Decontamination: Remove overlap with evaluation benchmarks
1.8.3
Training Configuration
from trl import
SFTTrainer , SFTConfig
sft_config = SFTConfig(
output_dir="./ sft_output",
max_seq_length =4096 ,
packing=True ,
# Pack
short
examples
into full
sequences
learning_rate =2e-5,
lr_scheduler_type ="cosine",
warmup_ratio =0.1,
weight_decay =0.01 ,
max_grad_norm =1.0,
num_train_epochs =3,
per_device_train_batch_size =4,
gradient_accumulation_steps =8,
bf16=True ,
gradient_checkpointing =True ,
)
trainer = SFTTrainer(model=model , args=sft_config ,
train_dataset =dataset , processing_class =tokenizer)
trainer.train ()
1.8.4
Efficient Training Solutions
Standard HuggingFace training leaves significant performance on the table. Several libraries provide
drop-in efficiency gains for SFT workloads:
Liger Kernel [88].
An open-source set of Triton-fused kernels from LinkedIn that replace
standard PyTorch operators during training. Key fusions include:
• Fused Cross-Entropy: Merges the final linear projection, softmax, and loss computation
into a single kernel—avoids materializing the full (batch × seq × vocab) logit tensor.
• Fused RMSNorm / SwiGLU / RoPE: Eliminates intermediate memory allocations for
common LLM building blocks.
• Chunked operations: Processes large tensors in tiles to keep peak memory bounded.
Result: 20% higher throughput and up to 60% memory reduction with a one-line integration
(apply_liger_kernel_to_llama()). Compatible with FSDP, DeepSpeed, and LoRA.
Unsloth [89].
A specialized fine-tuning library that combines custom CUDA/Triton kernels
with aggressive memory optimization:
• Manual backpropagation through LoRA layers (avoids autograd overhead).
• 4-bit QLoRA with fused dequantization—trains 70B models on a single 48 GB GPU.
• Intelligent RoPE and attention kernel fusion specific to each architecture (Llama, Mistral,
Qwen, Gemma).
Result: 2–5× faster than vanilla HuggingFace + PEFT, with 60–70% less VRAM. Particularly
impactful for single-GPU and consumer-hardware workflows.
72


<!-- page 73 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
torchtune [90].
Meta’s native PyTorch fine-tuning library (development wound down in 2025),
designed around composability rather than monolithic abstractions:
• Pure PyTorch—no trainer class; recipes are readable single-file scripts.
• Native integration with torch.compile, FSDP2, and activation checkpointing.
• First-class support for QLoRA, full fine-tuning, and knowledge distillation.
• Built-in quantization-aware training (QAT) for post-training compression.
Result: Comparable speed to custom solutions but with full debuggability and no framework
lock-in.
Choosing an Efficiency Stack
• Quick LoRA/QLoRA on ≤1 GPU: Unsloth (fastest time-to-train, minimal setup)
• Multi-GPU full fine-tune: TRL/DeepSpeed + Liger Kernel (best throughput at scale)
• Research / custom training loops: torchtune (transparent, hackable, native PyTorch)
These are complementary: Liger kernels can be used inside both TRL and torchtune workflows.
1.8.5
Best Practices
Table 1.11: SFT training guidelines.
Practice
Details
Packing
Concatenate multiple short examples into one sequence (separated
by EOS). Avoids padding waste.
NEFTune [91]
Add uniform noise to embeddings (α = 5). Improves MT-Bench
by 5–15% at zero cost.
Chat template
Always use the model’s native template. Mismatched templates
degrade quality.
Epochs
2–3 for large datasets; up to 5 for small (<10K) curated sets. Over-
training causes format memorization.
SFT Is Not Enough
SFT teaches format and basic instruction following, but cannot reliably teach: preference (which
response is better—needs RLHF/DPO), refusal (when not to answer—needs safety training),
calibration (saying “I don’t know”—needs RL with truthfulness rewards), or complex reasoning
(multi-step chains—needs RL with verifiable rewards). The full pipeline is: Pretrain →SFT →
RLHF/DPO.
1.9
LoRA and Parameter-Efficient Fine-Tuning
Full fine-tuning of a 70B model requires storing 70B trainable parameters plus their optimizer states
(560+ GB of memory). LoRA [92] (Low-Rank Adaptation) provides a way to fine-tune with <1% of
the parameters while achieving comparable quality.
1.9.1
The LoRA Insight
LoRA Core Idea
Instead of updating a full weight matrix W ∈Rd×d, learn a low-rank perturbation:
W ′ = W + α
r · BA,
B ∈Rd×r, A ∈Rr×d
73


<!-- page 74 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• W is frozen (no gradients, no optimizer states)
• Only B and A are trained: 2 × d × r parameters instead of d2
• At rank r = 16, d = 4096: LoRA adds 2 × 4096 × 16 = 131K params per layer vs. 16.8M for
full matrix
• α/r scaling controls the magnitude of the update
Why Low-Rank Works
Aghajanyan et al. [93] showed that fine-tuning operates in a very low-dimensional subspace — the
“intrinsic dimensionality” of the fine-tuning task is much smaller than the model’s parameter count.
A 175B model’s fine-tuning task may have intrinsic dimensionality <10,000. LoRA exploits this
directly: rank r constrains the update to an r-dimensional subspace per weight matrix.
Figure 1.10: LoRA decomposes the weight update ∆W into two small matrices B × A. The original weight
W remains frozen; only B and A receive gradients. At inference, the product BA can be merged into W with
zero overhead.
Why the α/r Scaling Matters
Without scaling, doubling the rank r would roughly double the magnitude of ∆W = BA (more
columns in B contribute to the sum). This means changing rank would also change how much the
model is perturbed—you’d need to re-tune the learning rate every time you adjust r.
The α/r factor normalizes the update magnitude so that it stays approximately constant
regardless of rank:
W ′ = W + α
r · BA
• Fix α, sweep r: The effective update magnitude stays ∼α regardless of rank. You can try
r ∈{8, 16, 32, 64} without re-tuning LR.
• Common practice: Set α = r (so α/r = 1) or α = 2r (so α/r = 2). This is a convenient
default where the scaling factor is a small integer.
• Why not just tune LR? You could, but α/r provides a rank-independent knob. Teams
can share LR recipes across experiments with different ranks.
• rsLoRA insight [94]: At high ranks (r ≥64), empirical evidence shows α/√r is more
stable than α/r, because the variance of BA scales with √r, not r.
74


<!-- page 75 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.9.2
LoRA Hyperparameters
Choosing LoRA hyperparameters correctly is critical — the wrong rank or alpha can either under-fit
(too constrained) or waste memory (too expressive).
Table 1.12: LoRA hyperparameter guide.
Hyperparameter
Typical Values
Guidance
r (rank)
8, 16, 32, 64
Higher = more capacity but more mem-
ory. Start with 16.
lora_alpha
16, 32 (often = r or 2r)
Controls update magnitude via α/r scal-
ing.
target_modules
q_proj, k_proj, v_proj,
o_proj
All
attention
projections.
Add
gate_proj, up_proj, down_proj for
full coverage.
lora_dropout
0.0–0.1
Regularization. Usually 0.05 for small
datasets.
bias
"none"
Training biases adds minimal params
but rarely helps.
Learning rate
1e-4 to 3e-4
Higher
than
full
fine-tuning
(only
adapters update).
Rank Selection Rules of Thumb
• r=8: Simple tasks (single-domain chat, classification). Very memory-efficient.
• r=16: General-purpose fine-tuning. Good default.
• r=32–64: Complex tasks (math, code, multi-turn reasoning). Approaches full fine-tune
quality.
• r=128+: Diminishing returns; consider full fine-tuning or QLoRA with higher rank.
• Key indicator: If training loss plateaus well above full fine-tune loss, increase rank.
1.9.3
LoRA Variants
Table 1.13: LoRA variants and their innovations.
Method
Key Innovation
When to Use
QLoRA [95]
4-bit quantized base + LoRA
in BF16.
NF4 data type +
double quantization.
Fine-tune 70B on single 48GB GPU.
DoRA [96]
Decomposes W into magni-
tude and direction; LoRA up-
dates direction only.
Better generalization for reasoning.
LoRA+ [97]
Different LRs for A/B (ηB =
ληA, λ ≈16).
Free 2% gain; no extra cost.
AdaLoRA [98]
Dynamic rank budget across
layers
(SVD-based
impor-
tance).
Very tight compute budget.
rsLoRA [94]
Scales by α/√r instead of α/r.
Stable at high ranks.
When using r ≥64.
VeRA [99]
Shared frozen random A, B;
trains diagonal scaling only.
Extreme param efficiency.
LoRA-FA
Freezes A after init; only trains
B. Halves LoRA memory.
Memory-constrained scenarios.
75


<!-- page 76 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Key Extensions Explained
DoRA – Weight-Decomposed Low-Rank Adaptation.
DoRA [96] observes that full fine-
tuning tends to change the direction of weight vectors more than their magnitude. Standard LoRA
conflates both. DoRA decomposes each weight column into magnitude m = ∥W∥col and direction
ˆV = W/∥W∥col, then applies LoRA only to the direction:
W ′ = m ⊙ˆV ′,
ˆV ′ =
W + BA
∥W + BA∥col
Magnitude m is a separate learnable vector (one scalar per column). This consistently outperforms
LoRA by 1–3% on reasoning and instruction-following benchmarks with no additional inference cost
(merged at deployment).
76


<!-- page 77 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
LoRA+ – Asymmetric Learning Rates.
Hayou et al. [97] show that matrices A and B in
LoRA have different optimal learning rates. Since B is initialized to zero, it starts in a very different
regime than A (initialized from N(0, σ2)). Setting ηB ≈16 × ηA improves convergence speed and
final quality by ∼2% — a free gain requiring only a one-line config change:
# LoRA+ in PEFT: set
different
LRs per matrix
optimizer_grouped_parameters = [
{"params": [p for n, p in model. named_parameters () if "lora_B" in n],
"lr": 2e-4 * 16},
# B matrix: higher LR
{"params": [p for n, p in model. named_parameters () if "lora_A" in n],
"lr": 2e-4},
# A matrix: base LR
]
VeRA – Vector-based Random Matrix Adaptation.
VeRA [99] takes parameter efficiency
to the extreme: instead of learning A and B, it freezes them as shared random matrices across all
layers and only trains two diagonal scaling vectors db ∈Rr and da ∈Rd:
∆W = B · diag(db) · A · diag(da)
This reduces trainable parameters by ∼10× vs. LoRA (only r + d params per layer) while achieving
90–95% of LoRA quality. Best for scenarios where you need hundreds of task-specific adapters with
minimal storage.
QLoRA Memory Savings
70B model full fine-tune: 140 GB (weights) + 280 GB (optimizer) + 140 GB (gradients) =
560 GB (7× A100-80GB).
70B QLoRA (r=16, all linear layers):
• Base model in NF4: 70B × 0.5 = 35 GB
• LoRA adapters in BF16: ∼160 MB
• Optimizer states (only for adapters): ∼320 MB
• Activations (gradient checkpointing): ∼8 GB
• Total: ∼44 GB — fits in a single 48GB GPU!
# QLoRA
configuration
with PEFT
from peft
import
LoraConfig , get_peft_model , prepare_model_for_kbit_training
from
transformers
import
BitsAndBytesConfig
import
torch
# 4-bit
quantization
config
bnb_config = BitsAndBytesConfig (
load_in_4bit=True ,
bnb_4bit_quant_type ="nf4",
# NormalFloat4 - optimal
for
weights
bnb_4bit_compute_dtype =torch.bfloat16 , # Compute in BF16
bnb_4bit_use_double_quant =True ,
# Quantize
the
quantization
constants
)
# LoRA
config
lora_config = LoraConfig(
r=16,
lora_alpha =32,
# alpha/r = 2x scaling
target_modules =["q_proj", "k_proj", "v_proj", "o_proj",
"gate_proj", "up_proj", "down_proj"],
lora_dropout =0.05 ,
bias="none",
task_type="CAUSAL_LM",
)
77


<!-- page 78 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
model = prepare_model_for_kbit_training (model)
# Prepare
for QLoRA
model = get_peft_model(model , lora_config)
# Add LoRA
adapters
model. print_trainable_parameters ()
# Output: trainable
params: 83 ,886 ,080 || all params: 70 ,553 ,706 ,496 || 0.12%
1.9.4
Other PEFT Approaches
LoRA dominates modern practice, but it is not the only parameter-efficient method. For completeness,
the main alternatives:
Table 1.14: PEFT method families. LoRA is the de facto standard for LLM fine-tuning; the others are
included for historical context and niche use cases.
Method
Mechanism
Pros / Cons
Status
LoRA [92] (and variants)
Low-rank
matrices
added
to
existing
weights
Mergeable at infer-
ence (zero overhead);
well-supported;
works for all architec-
tures
Standard
Adapters [100]
Small
bottleneck
MLPs
inserted
be-
tween layers
Modular;
stack-
able; adds inference
latency
(extra
se-
quential layers)
Rarely used
Prefix Tuning [101]
Learnable “virtual to-
kens” prepended to
keys/values at each
layer
No weight modifica-
tion; effective for gen-
eration tasks;
con-
sumes context length
Niche
Prompt Tuning [102]
Learnable
soft
prompt embeddings
prepended to input
Extremely
few
params
(<0.01%);
weaker than LoRA
for complex tasks
Niche
IA3 [103]
Learned vectors that
rescale keys, values,
and FFN activations
Even fewer params
than LoRA; merge-
able; limited capacity
Deprecated
BitFit [104]
Train only bias terms
Near-zero
params;
surprisingly effective
for simple tasks; lim-
ited expressiveness
Historical
Why LoRA Won
LoRA became the standard because it uniquely combines: (1) zero inference overhead —
adapters merge into base weights, unlike Adapters or Prefix Tuning which add latency or consume
context; (2) composability — multiple LoRA adapters can be swapped at serving time for
multi-tenant deployments; (3) ecosystem support — HuggingFace PEFT, TRL, vLLM, and
all major frameworks have first-class LoRA support; (4) proven at scale — used in production
by Meta, Google, and most open-source fine-tunes on HuggingFace. Unless you have a specific
constraint that LoRA cannot satisfy, it should be your default choice.
78


<!-- page 79 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.10
Mixture of Experts (MoE)
Mixture of Experts models [105, 106] scale model capacity without proportionally scaling compute
cost by activating only a subset of parameters for each token.
1.10.1
Architecture
MoE Layer
In a MoE transformer, the FFN layer in each block is replaced by N parallel “expert” FFNs plus
a router that selects which experts to use:
MoE(x) =
N
X
i=1
gi(x) · Ei(x),
g(x) = TopK(softmax(Wrx))
• Ei are expert networks (standard FFN layers)
• gi(x) are gating weights from the router (only top-K are non-zero)
• Typically K = 2 out of N = 8–64 experts are active per token
• Total params scale with N; active params scale with K/N of FFN size
Figure 1.11: MoE layer with 8 experts and Top-2 routing. Only the two highest-gated experts are computed
per token; the rest are skipped entirely.
1.10.2
Load Balancing
The Load Balancing Problem
Without constraints, the router may send most tokens to the same 1–2 experts (“expert collapse”).
This wastes capacity and creates compute imbalance across GPUs (each expert typically lives on
a different GPU).
Solution: Add an auxiliary load-balancing loss:
Lbal = α · N
N
X
i=1
fi · pi
where fi = fraction of tokens routed to expert i, pi = mean router probability for expert i. This
encourages uniform expert utilization.
79


<!-- page 80 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.10.3
Noisy Top-K Gating: Making Discrete Routing Trainable
The core challenge in MoE is that top-k selection is not differentiable — you can’t backpropagate
through a hard “pick the top 2” operation. The field has developed two key tricks to solve this:
The Routing Differentiability Problem
The router computes logits h(x) = Wr · x for each expert, then selects the top-k. But:
• The selected experts get gradients through their gate weights (softmax over selected)
• The selection decision itself (which k to pick) has zero gradient
• Without a trick, the router can get stuck: an expert never selected →never gets a gradient
signal →never gets selected
Approach 1: Noisy Top-K Gating [105].
Add learnable Gaussian noise to the router logits
before the top-k selection:
h(x) = Wg · x
(clean logits)
H(x) = h(x) + ϵ · Softplus(Wnoise · x),
ϵ ∼N(0, 1)
(noisy logits)
TopK(v, k)i =
(
vi
if vi is in the top k
−∞
otherwise
(1.11)
g(x) = softmax
 TopK(H(x), k)

(sparse gates)
• Wnoise is a learned noise magnitude — the model learns how much exploration each expert
needs
• During training, noise occasionally promotes “underdog” experts into the top-k, giving them
gradient signal
• At inference, noise is removed: use clean logits h(x) for deterministic routing
• The Softplus ensures noise scale is always positive
Approach 2: Gumbel-Softmax Trick (for differentiable discrete sampling).
An alternative
from the variational inference literature [107]. The Gumbel-Max trick provides exact sampling
from a categorical distribution:
z = arg max
i
[log πi + Gi] ,
Gi ∼Gumbel(0, 1)
(1.12)
where Gumbel noise is generated as Gi = −log(−log(Ui)), Ui ∼Uniform(0, 1).
For top-k routing: taking the top-k of (log πi + Gi) gives k samples without replacement from
the categorical distribution defined by π.
Since arg max is non-differentiable, the Gumbel-Softmax relaxation replaces it with a temperature-
controlled softmax:
ˆgi =
exp ((log πi + Gi)/τ)
P
j exp ((log πj + Gj)/τ)
(1.13)
• τ →0: approaches a hard one-hot (exact but non-differentiable)
• τ →∞: approaches uniform (differentiable but uninformative)
• In practice, anneal τ from 1.0 down to 0.1–0.5 during training
• Straight-through estimator: use hard top-k in the forward pass, but Gumbel-Softmax
gradients in the backward pass — best of both worlds
80


<!-- page 81 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Which Approach Is Used in Practice?
• Sparsely-Gated MoE [105], Mixtral [106], DeepSeek-V2 [108]: Use Noisy Top-K
with Gaussian noise. Simple, effective, well-proven at scale.
• Switch Transformer [109]: Simplified to Top-1 with no noise (relies on load-balancing
loss alone).
• Research / smaller-scale MoE: Some use Gumbel-Softmax for fully differentiable routing,
especially when learning the routing itself is the research objective.
• Key insight: Both approaches solve the same problem (making discrete selection trainable)
via noise injection.
Gaussian noise is simpler; Gumbel noise has stronger theoretical
guarantees for categorical sampling.
1.10.4
Notable MoE Models
Model
Total Params
Active Params
Experts
Innovation
Switch Transformer [109]
1.6T
100B
128, Top-1
First large-scale
MoE; simplified
routing
Mixtral 8x7B [106]
47B
13B
8, Top-2
Open-weight;
matches Llama-2
70B quality
DeepSeek-V2 [108]
236B
21B
160, Top-6
DeepSeekMoE
with
shared
+
routed experts
Qwen-MoE [32]
14.3B
2.7B
60, Top-4
Fine-grained
experts
for
efficiency
DBRX [110]
132B
36B
16, Top-4
Fine-grained
with 4 experts
per block
1.11
Diversity in LLM Training
Diversity — in training data, model outputs, and optimization trajectories — is critical for preventing
mode collapse and ensuring robust, general-purpose LLMs. This section covers the key diversity
mechanisms applicable to all LLM training phases.
1.11.1
Sampling Diversity
Sampling Strategies for Diverse Generation
• Temperature τ: P(xi) ∝exp(logiti/τ). Higher τ = more uniform distribution = more
diverse. Typical: τ = 0.7–1.0 for RLHF generation.
• Top-k: Only sample from the k highest-probability tokens.
Prevents degenerate low-
probability tokens.
• Top-p (nucleus): Sample from the smallest set of tokens whose cumulative probability ≥p.
Adaptive: more diverse when the model is uncertain.
• Min-p: Only keep tokens with P ≥pmin × Pmax. More principled than top-k.
• Frequency/presence penalty: Penalize tokens that appeared in the response. Encourages
lexical diversity.
81


<!-- page 82 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.11.2
Training Data Diversity
• Prompt diversity: Cover different domains, difficulty levels, and formats. The Goldilocks
principle: prompts should have 20–80% success rate.
• Deduplication: Remove near-duplicate training examples (MinHash, n-gram overlap). Dupli-
cates cause overfitting to specific patterns.
• Data mixing: Balance across tasks/domains using temperature-weighted sampling or curricu-
lum strategies.
1.11.3
Diversity-Promoting Methods
Method
How It Promotes Diversity
Temperature scaling
Higher τ flattens the distribution; more tokens become plausible.
Top-p / Min-p
Adaptive thresholds allow wider sampling when the model is un-
certain.
Frequency penalty
Penalizes repeated tokens, forcing lexical variety within a response.
Data deduplication
Removing near-duplicates from training data prevents overfitting
to specific patterns.
Multi-domain mixing
Temperature-weighted sampling across domains ensures broad
coverage.
Verbalized sampling
Prompt the model to explicitly verbalize a probability distribution
over responses [111]. See §7.5.
1.12
Text Generation: Decoding Methods
A trained language model outputs a probability distribution over the vocabulary at each step:
P(xt|x<t). The decoding strategy determines how we select the next token from this distribution.
This choice profoundly affects output quality, diversity, and coherence.
1.12.1
Greedy Decoding
The simplest strategy: always pick the highest-probability token.
xt = arg max
v∈V P(v|x<t)
Intuition: Like always taking the most obvious next word in a sentence. “The capital of France
is...” →“Paris” (probability 0.92).
Pros: Deterministic, fast, no hyperparameters.
Cons: Produces repetitive, generic text. Misses high-quality sequences where an early low-
probability token leads to a globally better output. No diversity.
1.12.2
Beam Search
Maintain B (beam width) partial hypotheses in parallel, expanding each by the top-k tokens and
keeping the B highest-scoring complete sequences:
score(y1:t) =
t
X
i=1
log P(yi|y<i)
With length normalization to avoid favoring short sequences:
scorenorm(y) =
1
|y|α
|y|
X
i=1
log P(yi|y<i),
α ∈[0.6, 1.0]
82


<!-- page 83 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Intuition: Like exploring multiple paths in a maze simultaneously, keeping only the B most
promising ones at each junction.
Pros: Finds higher-likelihood sequences than greedy; good for translation and summarization
where there’s a single “correct” output.
Cons: Still tends toward generic, repetitive text for open-ended generation; B× more compute;
all beams often converge to similar outputs.
Figure 1.12: Beam search with B = 2. At each step, only the 2 highest-scoring partial sequences survive
(blue). Lower-scoring alternatives are pruned (gray).
1.12.3
Diverse Beam Search
Standard beam search produces near-duplicate beams. Diverse beam search [112] partitions beams
into G groups and adds a dissimilarity penalty between groups:
scoreg(yt) = log P(yt|y<t) −λ
X
g′<g
∆(yt, Yg′)
where ∆measures overlap (e.g., Hamming diversity) with tokens already selected by earlier groups,
and λ controls diversity strength.
Intuition: Like forcing a brainstorming group to generate different ideas — each subgroup is
penalized for repeating what earlier subgroups said.
Pros: Produces genuinely different candidate sequences; useful for reranking pipelines.
Cons: Diversity penalty can degrade individual beam quality; more hyperparameters (G, λ).
1.12.4
Top-k Sampling
Sample from only the k most probable tokens, redistributing probability mass:
P ′(v|x<t) =





P(v|x<t)
P
v′∈Top-k P(v′|x<t)
if v ∈Top-k
0
otherwise
Intuition: After “The cat sat on the...”, only consider the top k plausible continuations (“mat”,
“floor”, “couch”, ...) and ignore extremely unlikely ones (“quantum”, “archipelago”).
Pros: Removes tail noise; simple to implement.
Cons: Fixed k is too restrictive for peaked distributions (wastes probability mass) and too
permissive for flat distributions (lets in garbage tokens).
83


<!-- page 84 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.12.5
Top-p (Nucleus) Sampling
Sample from the smallest set of tokens whose cumulative probability exceeds p:
Top-p = min
(
S ⊆V :
X
v∈S
P(v|x<t) ≥p
)
where tokens are sorted by descending probability and added until the threshold p is reached.
Intuition: Adaptively resize the candidate pool. If the model is confident (“Paris” at 95%), the
nucleus is tiny. If uncertain (“The movie was...”), the nucleus expands to include many plausible
adjectives.
Pros: Adapts to distribution shape; widely used default (p = 0.9–0.95).
Cons: Still includes some low-quality tokens at the tail of the nucleus; the threshold is a single
global hyperparameter.
Figure 1.13: Top-p (nucleus) sampling: tokens are sorted by probability and included until cumulative mass
reaches p = 0.9. The nucleus (dark blue) adapts its size to the distribution shape — here 5 tokens suffice.
Top-kk vs. Top-pp
Consider predicting the next word:
• After “2 + 2 =”: distribution is peaked — top-1 token (“4”) has 99% mass. Top-k=50
wastefully considers 49 wrong answers. Top-p=0.9 correctly picks just “4”.
• After “I enjoy eating”: distribution is flat — many foods are plausible. Top-k=5 is too
restrictive. Top-p=0.9 might include 50+ tokens, matching the actual uncertainty.
Top-p adapts; top-k doesn’t. In practice, both are often combined: sample from top-p intersected
with top-k.
1.12.6
Min-p Sampling
A recent alternative that sets a relative probability floor [113]:
Min-p =

v ∈V : P(v|x<t) ≥pmin · max
v′
P(v′|x<t)

Only tokens with probability at least pmin times the top token’s probability are kept.
Intuition: “Only consider tokens that are at least 10% as likely as the best token.” If the top
token has probability 0.8, only tokens above 0.08 survive. If the top token has probability 0.05 (very
uncertain), tokens above 0.005 survive — naturally expanding the pool.
Pros: Scales naturally with model confidence; fewer degenerate samples than top-p on peaked
distributions; single intuitive parameter.
84


<!-- page 85 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Cons: Newer, less battle-tested; not yet standard in all inference frameworks.
1.12.7
Temperature Scaling
Before applying any sampling strategy, logits are divided by temperature T:
PT (v|x<t) =
exp(zv/T)
P
v′ exp(zv′/T)
• T < 1: Sharpens distribution →more deterministic, focused outputs.
• T = 1: Unmodified model distribution.
• T > 1: Flattens distribution →more random, creative outputs.
• T →0: Becomes greedy decoding. T →∞: Becomes uniform sampling.
Common settings: T = 0.7 for factual tasks, T = 1.0–1.2 for creative writing, T = 0.0 (greedy)
for code/math.
1.12.8
Contrastive Decoding
Contrastive decoding [114] exploits the difference between a strong model (expert) and a weak model
(amateur) to amplify the expert’s unique knowledge:
xt = arg
max
v∈V(x<t) [log Pexpert(v|x<t) −log Pamateur(v|x<t)]
where V(x<t) = {v : Pexpert(v|x<t) ≥α · maxv′ Pexpert(v′|x<t)} is an adaptive plausibility constraint.
Intuition: The amateur model captures generic, obvious patterns (common words, repetition).
Subtracting its log-probabilities removes this “generic signal,” leaving the expert’s distinctive knowl-
edge and reasoning. Like removing background noise from a recording to hear the signal.
Pros: Reduces repetition and generic phrasing; improves factuality and coherence without addi-
tional training; works with any model pair.
Cons: Requires running two models (2× compute); sensitive to amateur model choice; the
plausibility threshold α needs tuning.
1.12.9
Repetition Penalties
Orthogonal to the sampling strategy, repetition penalties discourage the model from repeating tokens.
Given the raw logit zv for token v (i.e., the unnormalized score output by the LM head before
softmax), the penalized logit is:
z′
v =
(
zv/θ
if v ∈generated tokens and zv > 0
zv · θ
if v ∈generated tokens and zv < 0
where θ > 1 is the penalty factor (typically 1.1–1.3). In both cases, the effect is to push the logit
toward zero—reducing the probability of previously generated tokens. Frequency and presence
penalties are simpler additive variants used by OpenAI APIs:
z′
v = zv −α · count(v) −β · 1[v ∈generated]
where α is the frequency penalty (proportional to how many times v appeared) and β is the presence
penalty (flat penalty for any prior occurrence).
1.12.10
Practical Comparison
85


<!-- page 86 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Table 1.15: Decoding method comparison for LLM text generation.
Method
Deterministic
Diversity
Quality
Best For
Greedy
Yes
None
Medium
Code,
factual
QA
Beam Search (B=4–8)
Yes
Low
High (narrow)
Translation,
summarization
Diverse Beam Search
Yes
Medium
High
Candidate gener-
ation for rerank-
ing
Top-k (k=50)
No
Medium
Medium
General-purpose
generation
Top-p (p=0.9)
No
Adaptive
High
Default for open-
ended tasks
Min-p (pmin=0.1)
No
Adaptive
High
Robust alterna-
tive to top-p
Contrastive
Yes
Low
Very High
Factual,
coher-
ent long-form
Decoding in Practice: “Once upon a time”
Given the prompt “Once upon a time,”:
• Greedy: “there was a young girl who lived in a small village...” (generic fairy tale)
• Top-p=0.9, T=1.0: “the rivers ran backwards and the fish learned to fly...” (creative,
surprising)
• Top-p=0.9, T=0.3: “there was a kingdom ruled by a wise and just king...” (coherent,
conventional)
• Contrastive: “in the amber-lit corridors of a collapsing star, two minds argued about the
nature of time...” (distinctive, avoids clichés)
Same model, same prompt — decoding strategy determines the character of the output.
1.12.11
Constrained Decoding (Structured Generation)
All methods above sample from the full vocabulary at each step. Constrained decoding restricts
the set of allowed tokens so that the output is guaranteed to conform to a formal grammar—typically
a JSON schema, regex, or context-free grammar (CFG).
Core mechanism.
At each decoding step t, a token mask Mt ⊆V is computed from the current
parser state. Only tokens in Mt receive their original logits; all others are set to −∞before softmax:
P ′(v|x<t) =
(
P(v|x<t)/Z
if v ∈Mt
0
otherwise
where Z = P
v∈Mt P(v|x<t) renormalizes. Because the mask changes every step (it depends on what
has been generated so far), the constraint is enforced incrementally—the model cannot produce an
invalid prefix at any point.
From schema to mask.
The compilation pipeline is:
JSON Schema
compile
−−−−→Regex
compile
−−−−→FSM (DFA)
index
−−−→Token Mask per State
The FSM states correspond to positions in the regex. For each state, all vocabulary tokens that
would keep the string in the language are precomputed into an index (a one-time cost per schema).
86

