<!-- page 527 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Bottleneck analysis: Generation takes ∼45s for 128 responses. Training takes ∼15s. Scoring
takes ∼5s. Generation is bottleneck. Could move 8 GPUs from training to gen (40 gen, 24 training)
to balance, but then training is bottleneck. Current allocation is near-optimal.
Alternative if memory-tight: Move scoring onto training nodes (time-share: score during gen,
train during training). Saves 8 GPUs. Slightly worse pipelining but works.
Review: Chapters 2 and 11 (Systems Foundations; System Architecture).
27.5
GRPO Variants and Advanced RL Questions
Q21: What is DAPO and how does it improve over standard GRPO?
Answer: DAPO (Dynamic Adaptive Policy Optimization) introduces 5 key modifications:
1. Clip-Higher (asymmetric clipping): Standard PPO/GRPO clips both directions equally
at ϵ = 0.2. DAPO uses ϵlow = 0.2 but ϵhigh = 0.28. This allows the model to increase good action
probabilities more aggressively while still restricting how much it suppresses bad ones. Intuition:
exploration needs more room than exploitation.
2. Overlong Filtering: If a response hits the max length limit (truncated, no EOS token), it’s
masked entirely from the loss. Rationale: truncated responses contain no natural stopping signal
— training on them teaches the model that “stopping mid-sentence” is acceptable.
3. Token-level Loss: Loss is normalized by total token count across all sequences, not by number
of sequences. This prevents longer sequences from dominating the gradient.
4. Soft Overlong Punishment: Instead of binary truncation filtering, apply a gradual penalty
as responses approach max length. rsoft = −c · max(0, len −Lsoft)/(Lmax −Lsoft).
5. Dynamic Sampling: Resample prompts during training to ensure each batch has a mix of
success/failure (not yet in TRL).
When to use: Large-scale reasoning RL where you need maximum exploration and long comple-
tions (32K+ tokens). The asymmetric clipping is particularly valuable.
Review: Chapters 7 and 8 (GRPO; Preference Optimization Variants).
Q22: Explain the vLLM train-inference mismatch. Why does it happen and how do
TIS/MIS fix it?
Answer: The problem: When using vLLM for generation and a training framework (Deep-
Speed/FSDP) for updates, the same model with the same weights produces different token
probabilities. This happens because:
• Different numerical kernels (vLLM uses custom CUDA kernels optimized for throughput)
• Different attention implementations (Flash Attention in training vs PagedAttention in vLLM)
• Different precision handling (FP8/INT8 in vLLM vs BF16 in training)
• Batching differences affecting layer normalization numerics
This silently breaks PPO’s on-policy assumption: we compute the ratio πθ/πold using πold from
vLLM but πθ from the training framework. The ratio is wrong from step zero!
TIS
(Truncated
Importance
Sampling):
Correct the gradient by multiplying by
min(πtrain/πinference, C). The min with cap C prevents extreme corrections from destabilizing
training. Typical C = 2.0.
MIS (Masked Importance Sampling): More aggressive — simply discard any token where
πtrain/πinference > C. Zero contribution to gradient. Prevents any badly-estimated token from
affecting the update.
Sequence vs Token level: Sequence-level IS is theoretically correct (unbiased); token-level IS is
biased but lower variance. In practice, sequence-level with truncation works best.
Review: Chapters 7 and 11 (GRPO; System Architecture).
527


<!-- page 528 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q23: GSPO vs GRPO — what’s the fundamental difference and when does it matter?
Answer: GRPO: Computes importance ratio per token: wi,t = πθ(oi,t|q, oi,<t)/πold(oi,t|q, oi,<t),
then clips each token independently.
GSPO: Computes importance ratio at the sequence level: si(θ) = (πθ(oi|q)/πold(oi|q))1/|oi| — the
geometric mean of token probabilities. Clips this single sequence-level ratio.
Why it matters: GRPO’s per-token clipping treats each token as independent, but in language
they’re deeply correlated. A small per-token change early in the sequence compounds exponentially
over many tokens. GSPO captures this by looking at the full sequence probability.
Length normalization: The 1/|oi| exponent ensures fair comparison across different-length
sequences. Without it, longer sequences would always have lower probability ratios.
When
to
use
GSPO: When training goes off-policy (steps_per_generation > 1 or
num_iterations > 1). If fully on-policy (ratio ≈1), GRPO and GSPO are equivalent.
Review: Chapters 7 and 8 (GRPO; Preference Optimization Variants).
Q24: The paper “It Takes Two” shows G=2 matches G=16. How is that possible?
Answer: The key insight is that GRPO’s effectiveness doesn’t come from accurate advantage
estimation (which would need large G), but from an implicit contrastive objective.
With G = 2 and binary rewards (one correct, one wrong):
ˆAcorrect = +1, ˆAwrong = −1 (after
normalization). The loss becomes: increase probability of correct response, decrease probability of
wrong response. This is essentially a DPO-style contrastive loss!
Why large G doesn’t help much: The normalized advantage ˆAi = (ri −µ)/σ already creates a
contrast between good and bad. More samples give a better µ estimate, but the gradient direction
is dominated by the contrast between best and worst, not the mean accuracy.
Compute savings: G = 2 means 8× less generation compute than G = 16. Since generation is
60% of training time, this translates to ∼4× faster training.
Caveat: Works best when pass rate is 30–70%. If pass rate is very low (<10%), G = 2 often gives
two failures (no signal). Need larger G for hard problems.
Review: Chapter 7 (GRPO).
Q25: What is SAPO and how does its soft gating differ from hard clipping?
Answer: Standard PPO/GRPO uses hard clipping: clip(r, 1 −ϵ, 1 + ϵ). At the boundary, the
gradient suddenly drops to zero. This creates a “dead zone” where the model receives no learning
signal.
SAPO replaces this with a smooth sigmoid gate: the gradient is gradually attenuated as the ratio
moves away from 1, never suddenly zeroed out. It uses asymmetric temperatures:
• τ+ = 1.0 for positive advantages (standard attenuation)
• τ−= 1.05 for negative advantages (slightly more aggressive attenuation for suppression)
Benefits: (1) No “cliff” in gradient landscape. (2) Tokens slightly outside the clip range still
contribute (attenuated, not zeroed). (3) More stable optimization trajectory. (4) Sequence-coherent
— considers the full sequence context.
Trade-off: Slightly less restrictive trust region than hard clipping, so requires careful temperature
tuning. But more robust to hyperparameter choices overall.
Review: Chapters 7 and 8 (GRPO; Preference Optimization Variants).
528


<!-- page 529 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
27.6
DPO Extensions Questions
Q26: Compare f-DPO divergence choices. When would you use forward KL vs JS vs
reverse KL?
Answer: Standard DPO uses reverse KL implicitly (DKL[πθ∥πref]):
• Reverse KL (default): Mode-seeking. The policy concentrates probability where the refer-
ence is high. Avoids generating text the reference wouldn’t. Good for safety (conservative).
• Forward KL: Mass-covering. The policy tries to cover all modes of the reference, even
low-probability ones. Good for diversity but can generate low-quality outputs.
• Jensen-Shannon: Symmetric compromise between forward and reverse. Balanced mode-
coverage and mode-seeking. Often best for general alignment.
• Alpha-divergence (α = 0.5): Interpolates between forward (α = 0) and reverse (α = 1).
Tunable.
Practical recommendation: Start with reverse KL (standard DPO). If the model is too
conservative (won’t try creative solutions), switch to JS divergence. If diversity is critical (creative
writing, brainstorming), try forward KL.
Review: Chapters 6 and 8 (DPO; Preference Optimization Variants).
Q27: Your DPO preference data has 15% label noise. What do you do?
Answer: Three options in order of sophistication:
1.
Robust DPO (best for known noise rate): Analytically debiases the loss: Lrobust =
(1−ε)LDPO(yw,yl)−εLDPO(yl,yw)
1−2ε
.
Set ε = 0.15.
This provably recovers the clean DPO objective
in expectation. TRL: loss_type="robust", label_smoothing=0.15.
2. IPO (best when noise rate unknown): Squared loss with target margin. Mislabeled pairs have
bounded influence (the squared loss doesn’t diverge). More robust to arbitrary noise patterns
without needing to know ε. TRL: loss_type="ipo".
3. TR-DPO (best for distribution shift): Updates the reference model via EMA during train-
ing.
Even if early data is noisy, the evolving reference helps the model self-correct.
TRL:
sync_ref_model=True, ref_model_mixup_alpha=0.6.
Data-side fixes: (1) Filter pairs with <70% inter-annotator agreement. (2) Use RM to score
pairs; discard those where RM disagrees with label. (3) Active learning: re-label the most uncertain
pairs.
Review: Chapters 6 and 8 (DPO; Preference Optimization Variants).
Q28: What is SimPO and why is being reference-free advantageous?
Answer: SimPO uses the average log-probability of a response as an implicit reward signal:
r(x, y) =
1
|y|
P
t log πθ(yt|x, y<t) — no reference model needed.
The loss adds a target margin γ: the chosen response should have average log-prob at least γ
higher than rejected.
Why reference-free matters:
1. Memory: No reference model = 70–140GB saved for 70B models. Can train larger models
on same hardware.
2. Simplicity: No need to manage/load/serve a second model copy.
3. No stale reference: DPO’s reference becomes increasingly irrelevant as training progresses.
SimPO doesn’t have this problem.
4. Length normalization built in: The 1/|y| naturally prevents length bias (DPO needs
explicit handling).
529


<!-- page 530 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Trade-off: Without a reference anchor, the model has more freedom to collapse or drift. The γ
margin and length normalization partially mitigate this, but SimPO can be less stable than DPO
for aggressive training.
Review: Chapter 8 (Preference Optimization Variants).
Q29: Explain Iterative RPO. Why combine DPO with NLL loss for reasoning?
Answer: Standard DPO for reasoning has a subtle failure mode: it learns to discriminate (assign
higher implicit reward to correct traces) but doesn’t necessarily learn to generate them.
Why: DPO’s gradient pushes chosen probability up and rejected probability down. But the
chosen response might be so different from what the model would generate that increasing its
probability doesn’t teach the model how to produce similar reasoning patterns.
RPO’s fix: Add a negative log-likelihood (NLL/SFT) loss on the chosen response: L = LDPO +
α · LNLL(yw).
The NLL term explicitly trains the model to generate the winning response step by step. The
DPO term ensures it also learns to avoid the losing response. Combined: the model learns both
“how to reason correctly” (NLL) and “what to avoid” (DPO).
Iterative: Generate responses →check correctness →create pairs →train with RPO →repeat.
Each iteration the model gets better at generating correct reasoning, creating higher-quality
training data for the next round.
TRL: loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0]
Review: Chapters 8 and 13 (Preference Optimization Variants; RL for Large Reasoning Models).
27.7
GPU Architecture and Hardware Questions
Q30: Explain the GPU memory hierarchy. Why does it matter for LLM inference?
Answer: From fastest to slowest:
1. Registers: Per-thread, ∼256 KB/SM. Instant access (0 cycles latency).
2. SRAM (Shared Memory): Per-SM, ∼192–228 KB/SM (A100: 164 KB configurable).
Bandwidth: ∼19 TB/s aggregate. Latency: ∼20 cycles.
3. L2 Cache: Shared across GPU, 40–60 MB (H100: 50 MB). Bandwidth: ∼5 TB/s. Latency:
∼200 cycles.
4. HBM: Main GPU memory, 80 GB (A100). Bandwidth: 2–3.35 TB/s. Latency: ∼400 cycles.
5. CPU DRAM: Via PCIe, 512 GB+. Bandwidth: 32–64 GB/s. Latency: ∼10K cycles.
Why it matters for LLMs: Autoregressive generation reads the full model weights (∼140 GB
for 70B) for every single token. At 2 TB/s HBM bandwidth, that’s 70ms just to stream the
weights. The actual computation (one matrix-vector multiply) takes only 0.5ms. The GPU is 99%
waiting for data.
Flash Attention exploits this: By keeping intermediate results (QK scores, softmax) in SRAM
(19 TB/s) instead of writing them to HBM (2 TB/s), it eliminates 90% of the memory traffic for
attention. The compute is the same, but HBM reads/writes drop 10×.
Review: Chapter 2 (Systems Foundations for LLMs).
Q31: How does Flash Attention work? What is the online softmax trick?
Answer: Problem: Standard attention materializes the n × n attention matrix in HBM. For
n = 8192: 81922 × 2 = 134 MB per head, 4.3 GB per layer with 32 heads. Must write to HBM
then read back for softmax and multiply — 3 full HBM round-trips.
Flash Attention solution: Never store the full n × n matrix. Process in tiles that fit in SRAM.
Algorithm:
530


<!-- page 531 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1. Split Q into blocks of size Br rows, K/V into blocks of Bc rows.
2. For each Q block: iterate over all K blocks, computing partial attention scores.
3. Online softmax trick: Maintain running max m and running sum ℓfor softmax normal-
ization. When processing a new K block, update: mnew = max(mold, max(scores)), rescale
previous accumulator by emold−mnew, add new contribution.
4. Output is accumulated incrementally — never needs the full n × n matrix.
Key insight: Softmax is normally a global operation (max and P over all elements). The online
trick decomposes it into local updates with a correction factor. Mathematically exact — no
approximation.
Result: Memory O(n) instead of O(n2). Speed 2–4× faster (fewer HBM accesses, more time in
SRAM).
Flash Attention 2: Better work partitioning across warps, reduces non-matmul FLOPs by 2×.
Flash Attention 3 (H100/Hopper): Uses Tensor Memory Accelerator (TMA) for async loads,
warp specialization (producer/consumer warps), FP8 support.
Review: Chapters 1 and 2 (LLM Architecture; Systems Foundations).
Q32: Explain PagedAttention. How does it solve the KV cache problem?
Answer: The problem: During generation, each sequence needs a KV cache (stores K and V
tensors for all previous tokens). For a 70B model: each token needs 2 × nlayers × dmodel × 2 bytes
= 2 × 80 × 8192 × 2 ≈2.5 MB per token. A 2048-token sequence: ∼5 GB of KV cache.
Traditional allocation:
Pre-allocate max_sequence_length for each active sequence.
If
max=2048 but average=500, you waste 75% of allocated memory. With 50 concurrent sequences,
that’s hundreds of GB wasted.
PagedAttention: Inspired by OS virtual memory:
1. KV cache is split into fixed-size blocks (pages), each holding KV for 16 tokens.
2. A block table (like a page table) maps logical token positions to physical memory blocks.
3. Blocks are allocated on demand as the sequence grows. No pre-allocation waste.
4. Freed blocks return to a pool immediately when sequences finish.
Extra benefits:
• Prefix sharing: Multiple sequences with the same system prompt share KV cache blocks
(copy-on-write). Saves 30–50% memory for chat applications.
• Preemption: Can “swap out” a low-priority sequence’s blocks to CPU, freeing GPU memory
for higher-priority requests.
• Near-zero fragmentation: Internal fragmentation limited to last block (<16 tokens).
External fragmentation eliminated (any free block can be used anywhere).
Result: 3–5× more concurrent sequences in the same memory →3–5× better throughput for
serving.
Review: Chapter 2 (Systems Foundations for LLMs).
Q33: Compare NVLink vs InfiniBand. When do you use each in RLHF training?
Answer:
NVLink (intra-node, GPU-to-GPU):
• Bandwidth: 600 GB/s (A100), 900 GB/s (H100) — total bidirectional
531


<!-- page 532 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Latency: ∼1 µs
• Scope: Within one physical node (8 GPUs connected via NVSwitch)
• Use case: Tensor Parallelism (TP=8). Each layer’s matrix multiply is split across GPUs,
requiring AllReduce after every layer. Needs ultra-high bandwidth + low latency.
InfiniBand NDR (inter-node, node-to-node):
• Bandwidth: 400 Gb/s = 50 GB/s per port. With 8 ports (GPUDirect RDMA): 400 GB/s
aggregate per node.
• Latency: ∼1–5 µs (RDMA)
• Scope: Between nodes in a cluster. Requires switches (fat-tree topology).
• Use case: Data Parallelism / FSDP gradient synchronization. AllReduce of gradients
happens once per training step (not per layer), so latency tolerance is higher.
In RLHF specifically:
• Generation: TP=8 over NVLink within node. Multiple vLLM instances across nodes don’t
communicate (embarrassingly parallel).
• Training: TP=8 over NVLink intra-node + FSDP over InfiniBand inter-node. Gradients
synced after full backward pass.
• Weight sync: Training →Generation uses InfiniBand (140 GB transfer, async, takes ∼3s at
50 GB/s).
Review: Chapters 2 and 11 (Systems Foundations; System Architecture).
27.8
Optimization and Training Questions
Q34: Explain Adam vs AdamW. Why does the difference matter for LLMs?
Answer: Adam with L2 regularization: θt+1 = θt −α · ( ˆmt/(√ˆvt + ϵ) + λθt). The weight
decay term λθt is inside the adaptive scaling. Parameters with large gradients (large vt) get less
weight decay (divided by √vt). This is not true weight decay — it’s scale-dependent.
AdamW (decoupled weight decay): θt+1 = (1 −αλ)θt −α · ˆmt/(√ˆvt + ϵ). Weight decay is
applied outside and before the adaptive update. Every parameter gets the same proportional decay
regardless of gradient history.
Why it matters for LLMs:
1. LLMs have parameters spanning many orders of magnitude (embedding layers vs attention
vs FFN). Adam’s coupled L2 effectively penalizes small-gradient params more than large-
gradient ones — wrong behavior.
2. Decoupled WD provides uniform regularization across all layers, preventing some layers from
growing unbounded while others over-shrink.
3. Empirically: AdamW gives 2–5% better perplexity on long pretraining runs compared to
Adam+L2 with the same effective regularization.
For RL specifically: Often use λ = 0 (no weight decay). The KL penalty provides regularization.
But for SFT: λ = 0.01–0.1 with AdamW is standard.
Review: Chapter 1 (LLM Architecture and Optimization Methods).
532


<!-- page 533 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q35: Why is learning rate warmup necessary? What happens without it?
Answer: The problem: Adam’s second moment estimate vt = β2vt−1 + (1 −β2)g2
t starts at
v0 = 0. The bias correction ˆvt = vt/(1 −βt
2) compensates mathematically, but in practice:
• First few steps: vt is based on 1–5 gradient samples. Highly inaccurate estimate of true
variance.
• If a parameter happens to get a small gradient initially, vt is tiny →effective LR is huge →
catastrophic update.
• The bias correction amplifies early updates: at step 1, ˆv1 = v1/(1 −0.999) = 1000 · v1.
Without warmup: First 10–100 steps often have gradient spikes that permanently damage the
model. Early representations get scrambled before the optimizer stabilizes.
Warmup fix: Start with LR ≈0 and linearly increase to target over W steps (typically 3–10% of
training). By the time LR reaches full value, vt has accumulated enough samples to be accurate.
Typical settings:
• Pretraining: 2000 steps warmup (∼1% of 200K steps)
• SFT: 100 steps warmup (∼5% of 2000 steps)
• RL (PPO/GRPO): 20–50 steps warmup (short, model already stable from SFT)
Review: Chapters 1 and 10 (LLM Architecture; SFT Best Practices).
Q36: Compare learning rate schedules. Which would you choose for RL fine-tuning?
Answer:
Cosine decay: ηt = ηmin + 1
2(ηmax −ηmin)(1 + cos(πt/T)). Standard for pretraining and SFT.
Smooth decay, most time at moderate LR.
Linear decay: ηt = ηmax(1 −t/T). Simpler, similar results to cosine for short runs.
WSD (Warmup-Stable-Decay): Warmup →constant LR for 80% →rapid decay in last
20%. New standard for pretraining. The “stable” phase gives consistent learning; the final decay
squeezes out remaining gains.
Constant: No decay. ηt = ηmax after warmup.
For RL fine-tuning (PPO/GRPO), I’d choose: Constant with short warmup. Reasons:
1. RL training length is highly unpredictable (you stop based on win-rate, not epochs).
2. Cosine/linear decay assumes you know the total steps in advance.
3. The LR is already very low (10−6), further decay makes updates invisible.
4. PPO’s adaptive KL controller already modulates the effective step size.
5. If you must decay: use linear decay over a generous budget, stop early when metrics plateau.
Review: Chapters 1 and 5 (LLM Architecture; PPO).
Q37: Why is gradient clipping critical for RL training but less important for SFT?
Answer: SFT: Supervised loss is smooth and well-behaved. Gradient norms are consistent across
batches (typically 0.1–1.0). Clipping at 1.0 rarely activates — it’s a safety net.
RL (PPO/GRPO): Gradient norms are highly variable because:
1. Reward variance: One batch might have all high-reward responses, next might have all
low. The advantage ˆA swings wildly.
2. Ratio explosion: If a rare token’s probability changed a lot, rt = πnew/πold can be very
533


<!-- page 534 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
large →large gradient before clipping kicks in.
3. Sparse reward: In GRPO with binary rewards, some prompts give all-correct (advantage
≈0) then suddenly a hard prompt gives extreme advantages.
4. KL term: The KL penalty gradient can spike when policy diverges.
Without clipping: A single bad batch can produce a gradient 100× normal magnitude →
destroys the model in one step. Recovery is impossible (catastrophic forgetting of all pretraining).
Typical setting: max_grad_norm=1.0. Some use 0.5 for extra safety in early RL training. The
norm is computed globally across all parameters (not per-layer).
Monitoring: If clipping activates more than 20% of steps, your LR is probably too high or your
batch size too small.
Review: Chapters 5 and 7 (PPO; GRPO).
Q38: BF16 vs FP16 for training. When does the choice matter?
Answer:
FP16: 1 sign + 5 exponent + 10 mantissa bits. Range: ±65504. Precision: ∼3.3 decimal digits.
BF16: 1 sign + 8 exponent + 7 mantissa bits. Range: ±3.4 × 1038 (same as FP32!). Precision:
∼2.4 decimal digits.
Why BF16 wins for LLMs:
1. No loss scaling needed: FP16’s small range (±65K) means gradients and activations
frequently overflow/underflow. Requires dynamic loss scaling (multiply loss by 1024, divide
gradients back). BF16 has FP32’s range — overflow is essentially impossible.
2. Simpler code: No loss scaler, no inf/nan checking, no dynamic scaling adjustment.
3. Critical for RL: RL gradients are noisier and spikier than SFT. FP16 loss scaling often
fails (picks wrong scale, causes NaN). BF16 “just works.”
When FP16 might be better: If you need maximum precision (some scientific computing tasks)
and can manage the loss scaling. FP16 has 3 more mantissa bits = slightly more accurate results.
FP32 master weights: Even with BF16 forward/backward, accumulate gradient updates in
FP32 to prevent rounding errors from compounding over thousands of small steps. Standard
practice for all LLM training.
Review: Chapter 2 (Systems Foundations for LLMs).
27.9
Reward Model and SFT Questions
Q39: Derive the Bradley-Terry reward model loss. What are its limitations?
Answer: Bradley-Terry model: Given two responses, the probability the better one (yw) is
preferred: P(yw ≻yl|x) = σ(r(x, yw) −r(x, yl)) where σ is the sigmoid.
MLE derivation: Given N preference pairs, maximize likelihood: Q
i P(yi
w ≻yi
l). Take negative
log: L = −P
i log σ(r(xi, yi
w) −r(xi, yi
l)).
Limitations:
1. No ties: BT can’t model “equally good” — forces a strict preference.
2. Transitivity: Assumes if A>B and B>C then A>C. Humans aren’t transitive.
3. Context-free: Same reward regardless of what alternatives were available.
4. Scalar collapse: Compresses all quality dimensions into one number. A response can be
safe but unhelpful — RM must trade off.
534


<!-- page 535 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
5. Length bias: Longer responses get higher scores (more information = more likely to contain
what annotator wanted). Must explicitly decorrelate.
Mitigations: Margin loss (require minimum gap δ), reward centering (subtract running mean),
length penalty during training, multi-head RM (separate scores for helpfulness/safety/accuracy).
Review: Chapter 9 (Reward Model Training).
Q40: What is sequence packing in SFT and why does it matter?
Answer: Problem: Training examples have variable length. Standard batching pads all examples
to max_length. If max=4096 but average=500, you waste 88% of compute on padding tokens
(which contribute zero gradient).
Packing solution: Concatenate multiple short examples into a single max_length sequence.
Separate with EOS tokens. Train on all examples simultaneously.
Example: Instead of 4 sequences padded to 4096 (16K tokens, 14K padding), pack into 1 sequence
of 4096 with 4 examples end-to-end (4096 real tokens, 0 padding). 4× more efficient.
Critical detail — block-diagonal attention mask: Without special handling, example 2
attends to example 1’s tokens (cross-contamination). Must use a block-diagonal attention mask
that restricts each example to only attend to its own tokens.
In TRL: SFTConfig(packing=True, max_seq_length=4096). Handles mask automatically.
Caveats: (1) Longer examples still need their own batch entries (can’t split mid-sequence). (2)
Slight implementation complexity for position embeddings (reset per example). (3) Some argue
packing changes the effective batch size (more examples per step) — adjust LR accordingly.
Review: Chapter 10 (SFT Best Practices and Techniques).
Q41: Explain completion-only masking for SFT. What happens if you don’t use it?
Answer: In chat-format SFT data: [system] + [user message] + [assistant response].
Standard NLL loss computes loss on all tokens including the system prompt and user message.
Problem without masking: The model wastes capacity learning to predict the user’s message
(which it will never need to generate at inference). Worse: if the training data has diverse user
messages, the model gets confused about “whose turn is it?”
Completion-only masking: Set loss weight to 0 for all tokens in the prompt (system + user).
Only compute loss on assistant response tokens.
TRL: DataCollatorForCompletionOnlyLM(response_template="<|assistant|>")
Impact: Typically 5–15% better on instruction-following benchmarks.
Faster convergence
(gradient signal is concentrated on useful tokens). No change to compute cost.
Subtlety: Must include the response template token in the loss (teaches the model to start
responding). But exclude everything before it.
Review: Chapter 10 (SFT Best Practices and Techniques).
Q42: How does SFT quality affect the RL ceiling? What is the pass@k diagnostic?
Answer: Ceiling theorem (informal): RL can only reinforce behaviors the model can already
produce with non-negligible probability. If the SFT model has 0% chance of generating a correct
solution, RL will never find it.
Why: GRPO/PPO sample from the current policy and reinforce good samples. If good samples
don’t exist in the distribution, there’s nothing to reinforce. RL is exploration-limited by the base
policy’s support.
pass@k diagnostic: Generate k responses per prompt, check if any is correct:
• pass@1: Model’s typical performance (greedy/low temp).
• pass@8: Upper bound of what GRPO with G = 8 can achieve.
• pass@64: Upper bound for aggressive Best-of-N.
• pass@256: Approximate ceiling for RL improvement.
535


<!-- page 536 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Interpretation:
• pass@1=20%, pass@64=80%: Great! 4× headroom for RL. Strong gains expected.
• pass@1=20%, pass@64=25%: Almost no headroom. RL won’t help much. Need better SFT
first.
• pass@1=5%, pass@64=60%: Model can solve it but rarely does. Perfect case for RL (reinforce
the rare successes).
Rule: If pass@64 < 1.5× pass@1, invest in better SFT data before starting RL.
Review: Chapters 7 and 10 (GRPO; SFT Best Practices).
Q43: Design a multi-objective reward system for a chat model. How do you balance
helpfulness vs safety?
Answer: Architecture: Separate reward models for each objective:
• rhelpful: Trained on helpfulness preferences (quality, accuracy, completeness)
• rsafe: Trained on safety preferences (refusals, harmlessness, no hallucination)
• rformat: Rule-based (follows instructions, proper formatting, appropriate length)
Combination strategies:
1. Weighted sum (simplest): r = w1rhelpful + w2rsafe + w3rformat. Problem: safety can be
outweighed by helpfulness.
2. Constrained (safer): Maximize rhelpful subject to rsafe > τ.
Implemented via: r =
rhelpful −λ · max(0, τ −rsafe) with large λ.
3. GDPO normalization (best for GRPO): Normalize each reward independently within
group, then combine: ˆA = w1 ˆAhelpful + w2 ˆAsafe. Prevents one reward from dominating due
to scale differences.
4. Lexicographic: Safety is hard constraint (must pass), then optimize helpfulness. Train in
stages: safety alignment first, then helpfulness.
Practical weights: Start with wsafe = 2.0, whelpful = 1.0, wformat = 0.5. Safety gets 2× weight
because its failure mode (harmful content) is much worse than helpfulness failure (mediocre
answer).
Review: Chapters 9 and 12 (Reward Model Training; LLM Agentic Training).
27.10
System Architecture Extension Questions
Q44: How does speculative decoding work? When does it help for RLHF?
Answer: Problem: Large model generates one token per forward pass (∼70ms for 70B). Slow.
Speculative decoding:
1. Draft: Small model (1–7B) generates k candidate tokens quickly (∼5ms for all k).
2. Verify: Large model does one forward pass scoring all k tokens in parallel. Accepts tokens
where plarge(ti) ≥pdraft(ti) (always). Probabilistically accepts others.
3. Result: On average, 3–4 tokens accepted per verification step. Speedup: 2–3×.
Key property: The output distribution is identical to sampling from the large model alone. No
quality loss. The draft model only affects speed, not output.
536


<!-- page 537 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
For RLHF specifically: Generation is 60% of compute. 2–3× speedup on generation = 1.5–2×
end-to-end speedup. Combined with vLLM + INT8: generation goes from the bottleneck to parity
with training.
Limitations: (1) Draft model must share tokenizer. (2) Less effective at high temperature (draft
model less accurate). (3) Needs additional GPU memory for draft model. (4) Diminishing returns
beyond k = 5 (acceptance rate drops).
Review: Chapters 2 and 11 (Systems Foundations; System Architecture).
Q45: Explain the roofline model. How do you determine if a kernel is compute-bound
or memory-bound?
Answer: The roofline model plots achievable performance (FLOPS) as a function of arithmetic
intensity (FLOPS per byte of memory traffic).
Two regimes:
• Memory-bound (left of crossover): Performance limited by how fast you can feed data
to compute units. Actual FLOPS = bandwidth × arithmetic intensity. GPU utilization <
100%.
• Compute-bound (right of crossover): Performance limited by peak FLOPS. Memory is
fast enough. GPU at max utilization.
Crossover point: Peak FLOPS / Peak Bandwidth.
For A100: 312 TF / 2 TB/s = 156
FLOP/byte.
LLM operations:
• Autoregressive generation (batch=1): Read 140GB weights, do 140G FLOPs = 1
FLOP/byte. Extremely memory-bound (156× below crossover). Only 0.6% GPU utilization.
• Training forward pass (batch=128, seq=2048): Arithmetic intensity ≈200+ FLOP/byte.
Compute-bound. Near peak utilization.
• Attention (long sequence): O(n2d) FLOPs / O(n2 +nd) bytes. For long n: compute-bound.
For short n: memory-bound. Flash Attention keeps it in SRAM regardless.
Practical use: If your kernel is memory-bound, reduce memory traffic (quantization, caching,
tiling). If compute-bound, reduce FLOPs (pruning, distillation, lower precision).
Review: Chapter 2 (Systems Foundations for LLMs).
Q46: How does continuous batching work and why is it essential for RLHF generation?
Answer: Static batching: Start B sequences. Wait for ALL to finish. If one sequence generates
500 tokens and another generates 50 tokens, the 50-token sequence’s GPU slot sits idle for 450
tokens.
Continuous batching (iteration-level scheduling): After each generation step, check which
sequences are done. Immediately insert new sequences into freed slots. GPU slots are never idle.
Why essential for RLHF:
1. RLHF generates diverse outputs (high temperature). Length variance is huge — some
responses are 50 tokens, others 2000+.
2. Without continuous batching: average utilization ∼40–50% (waiting for slowest sequence).
3. With continuous batching: utilization >90%. Throughput 2–3× higher.
4. RLHF needs large batches (128+ responses per step). Generating 128 responses with static
batching requires max_tokens × 128 sequential steps. Continuous batching amortizes this.
537


<!-- page 538 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Implementation: vLLM’s scheduler checks after every decode step. Preemption: if a new
high-priority request arrives and memory is full, can swap out a low-priority sequence’s KV cache
to CPU and resume later.
Review: Chapters 2 and 11 (Systems Foundations; System Architecture).
27.11
Transformer Architecture Questions
Q: Why does RoPE dominate over learned absolute positional embeddings in modern
LLMs?
Answer: RoPE encodes relative position directly into the Q/K dot product via rotation matrices.
Key advantages:
1. Attention scores depend only on relative distance i−j, not absolute position — this generalizes
better to unseen sequence lengths.
2. Can be extended beyond training length via frequency scaling (NTK-aware, YaRN) without
retraining.
3. No additional parameters (rotations are deterministic from position index).
4. Learned absolute embeddings are fixed to training length and don’t extrapolate — a model
trained at 4K context fails at 8K.
Review: Chapter 1 (LLM Architecture and Optimization Methods).
Q: Explain SwiGLU and why it replaced ReLU in modern transformers.
Answer: SwiGLU: FFN(x) = W2(Swish(W1x) ⊙W3x), where Swish(x) = x · σ(x).
Why it’s better:
• The gating mechanism (⊙W3x) allows the network to selectively suppress or amplify dimen-
sions — more expressive than pointwise ReLU.
• Swish is smooth (no dead neurons like ReLU’s zero-gradient region).
• Empirically: 1–2% improvement on language modeling benchmarks at same FLOP count.
• Tradeoff: requires 3 weight matrices instead of 2 (solved by reducing hidden dim from 4d to
8d/3).
Review: Chapter 1 (LLM Architecture and Optimization Methods).
Q: What is Grouped Query Attention (GQA) and why does Llama-3 use it?
Answer: Standard MHA: H query heads, H key heads, H value heads. GQA: H query heads
but only G < H key/value heads (shared across query groups).
Llama-3 70B: 64 query heads, 8 KV heads (each KV head shared by 8 query heads).
Benefits:
• KV cache size reduced by H/G = 8× — critical for inference (KV cache is the dominant
memory cost at long sequences).
• Minimal quality loss (<0.5% on benchmarks) because KV patterns are highly correlated
across heads.
• Inference throughput increases proportionally to KV cache reduction (more sequences fit in
memory = higher batch size).
Review: Chapter 1 (LLM Architecture and Optimization Methods).
538


<!-- page 539 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q: Why did decoder-only architectures win over encoder-decoder for LLMs?
Answer:
1. Unified objective: Pretraining = fine-tuning = inference all use next-token prediction. No
architectural mismatch.
2. Parameter efficiency: All parameters contribute to generation.
In encoder-decoder,
encoder params are “wasted” during pure generation tasks.
3. Simpler scaling: One model, one loss function, one set of hyperparameters to tune.
4. KV cache efficiency: Decoder-only has one KV cache; encoder-decoder has two (encoder
+ decoder cross-attention).
5. Emergent few-shot: Decoder-only naturally supports in-context learning (prepend exam-
ples to the prompt).
Encoder-decoder still wins for seq2seq tasks with fixed input length (translation), but these are a
shrinking fraction of LLM use cases.
Review: Chapter 1 (LLM Architecture and Optimization Methods).
27.12
Flash Attention Questions
Q: Flash Attention computes the same result as standard attention but is 2–4× faster.
How is this possible if it does the same number of FLOPs?
Answer: Flash Attention is faster because it reduces HBM memory traffic, not FLOPs. Standard
attention materializes the n × n attention matrix in HBM (slow), reads it back for softmax, reads
again for PV multiply — 4 HBM round-trips over O(n2) data.
Flash Attention tiles the computation so the n × n matrix is computed and consumed entirely
in SRAM (fast, 19 TB/s) without ever writing it to HBM (2 TB/s). The “online softmax” trick
enables this by maintaining running statistics.
Result: HBM traffic drops from O(n2d) to O(n2d/M) where M is SRAM size. Same FLOPs,
10–50× less memory traffic →2–4× wall-clock speedup.
Review: Chapters 1 and 2 (LLM Architecture; Systems Foundations).
Q: Why doesn’t Flash Attention help the FFN layers?
Answer: FFN layers are compute-bound, not memory-bound. Their arithmetic intensity (I ≈300
FLOP/byte for large batch GEMMs) is already above the roofline ridge point (156 FLOP/byte on
A100).
Flash Attention helps attention because attention is deeply memory-bound (I ≈1–60 FLOP/byte).
By keeping data in SRAM, it removes the memory bottleneck.
For FFN: the bottleneck is already the Tensor Cores (not memory bandwidth), so reducing
memory traffic doesn’t help. Instead, FFN benefits from quantization (reduces weight size →
higher arithmetic intensity) and larger batch sizes.
Review: Chapters 1 and 2 (LLM Architecture; Systems Foundations).
Q: Explain the online softmax trick and why it’s essential for Flash Attention.
Answer: Standard softmax needs the global maximum m = maxj xj before computing any output
— this requires seeing all n attention scores first, forcing materialization of the full n × n matrix.
The online softmax trick processes blocks sequentially, maintaining a running (m, ℓ, O) state:
1. Process new block →update running max: mnew = max(mold, max(snew))
2. Rescale old sum: ℓnew = emold−mnew · ℓold + new terms
539

