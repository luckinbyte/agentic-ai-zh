<!-- page 513 -->
Chapter 27
Quiz Questions & Detailed Answers
This chapter provides a comprehensive set of questions designed to test and reinforce your understand-
ing of the material covered throughout this guide. Each question targets a key concept, algorithm,
or system design decision—the kind of knowledge that distinguishes surface-level familiarity from
genuine expertise. Use these questions for self-assessment: attempt your own answer before reading
the detailed solution. The questions progress from foundational concepts (LLM architecture, rein-
forcement learning basics) through core algorithms (PPO, DPO, GRPO) to advanced system design
and agentic AI topics.
27.1
Foundations Questions
Q0a: What is the role of the attention mechanism in a decoder-only Transformer?
Why is it causal?
Answer: The attention mechanism allows each token to attend to (i.e., compute a weighted
combination of) representations from other tokens. In a decoder-only Transformer, attention is
causal (also called autoregressive): token t can only attend to tokens 1, . . . , t, never to future
tokens t + 1, . . . , T.
Why causal? Because the model generates text left-to-right. At inference time, future tokens
literally do not exist yet. The causal mask during training simulates this constraint so that the
model learns to predict each token using only its left context. Mathematically, the attention
matrix is masked:
Attention(Q, K, V ) = softmax
 
QK⊤
√dk
+ M
!
V
where Mij = −∞for j > i (future positions), forcing those attention weights to zero.
Practical implication: This enables the KV-cache optimization at inference—since past tokens’
keys and values never change, they can be cached and reused, reducing generation from O(T 2) to
O(T) per new token.
Review: Chapter 1 (LLM Architecture and Optimization Methods).
Q0b: Explain Flash Attention. What problem does it solve and how?
Answer: Standard attention computes the full T × T attention matrix, which requires O(T 2)
memory and is memory-bandwidth bound—the GPU spends most of its time moving data
between HBM (slow, large) and SRAM (fast, small), not doing actual computation.
Flash Attention’s insight: Never materialize the full attention matrix in HBM. Instead, tile
the computation into blocks that fit in SRAM, compute attention block by block using an online
softmax algorithm, and write only the final output to HBM.
Key techniques:
1. Tiling: Split Q, K, V into blocks of size Br × Bc that fit in SRAM
2. Online softmax: Track running max and sum to compute softmax incrementally without
513


<!-- page 514 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
the full row
3. Recomputation: In the backward pass, recompute attention from Q, K, V (cheap) rather
than storing the T × T matrix (expensive)
Result: O(T) HBM memory (instead of O(T 2)), 2–4× wall-clock speedup, exact same numerical
output (not an approximation).
Review: Chapters 1–2 (LLM Architecture; Systems Foundations).
Q0c: What is the difference between SFT, RLHF, and DPO at a high level? When
do you use each?
Answer:
• SFT (Supervised Fine-Tuning): Train the model to imitate high-quality demonstrations.
Loss: next-token prediction on curated data. Teaches format and style.
• RLHF: Train a reward model from human preferences, then optimize the policy against it
using RL (PPO). The model explores beyond the demonstration data. Teaches what humans
prefer.
• DPO: Skip the reward model. Directly optimize the policy on preference pairs (yw, yl) using
a contrastive loss. Same goal as RLHF but simpler pipeline.
Typical pipeline: SFT first (gives the model a good starting point), then RLHF or DPO (refines
preferences). SFT alone tends to produce verbose, hedge-heavy responses. RLHF/DPO makes
outputs more direct and aligned with human intent.
When to use each: SFT when you have gold-standard outputs. DPO when you have preference
pairs but limited compute. RLHF (PPO) when you need maximum quality and can afford the
infrastructure.
Review: Chapters 5, 6, and 10 (PPO; DPO; SFT Best Practices).
Q0d: What is a reward model? How is it trained and what can go wrong?
Answer: A reward model (RM) is a neural network that takes a (prompt, response) pair and
outputs a scalar score indicating quality. It is trained on human preference data: given pairs
(yw, yl) where yw is preferred, the RM learns to assign R(yw) > R(yl).
Training: Bradley-Terry loss: L = −log σ(R(yw) −R(yl)). Architecture: typically the same
transformer as the policy, with the LM head replaced by a scalar projection.
What can go wrong:
1. Reward hacking: The policy finds outputs that score high on the RM but are actually low
quality (e.g., excessively long, repetitive, or containing specific phrases the RM was biased
toward)
2. Distribution shift: The RM was trained on outputs from an earlier policy. As training
progresses, the current policy generates out-of-distribution outputs the RM cannot score
accurately
3. Label noise: Human annotators disagree, are tired, or apply inconsistent criteria. This
noise propagates into RM predictions
4. Overconfidence: The RM assigns extreme scores to outputs it has never seen, providing
misleading gradient signal
Review: Chapter 9 (Reward Model Training).
514


<!-- page 515 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q0e: Explain the exploration-exploitation trade-off in RL. How does it manifest in
LLM training?
Answer: In RL, an agent must balance:
• Exploitation: choosing actions known to yield high reward (greedy behavior)
• Exploration: trying new actions that might yield even higher reward (but might also fail)
In LLM training: The policy is the language model. “Actions” are token choices. “Exploitation”
means generating responses similar to what already scored well. “Exploration” means trying novel
phrasings, structures, or reasoning paths.
How it manifests:
• Temperature during generation: Higher temperature = more exploration. GRPO uses
temperature 1.0 to get diverse samples within each group.
• KL penalty: Acts as an anti-exploration brake—prevents the policy from straying too far
from the reference model. Without it, the policy might collapse to a single high-reward
template (mode collapse).
• Group sampling in GRPO: Generating G responses per prompt explicitly explores the
output space, then reinforces above-average responses.
The tension: Too little exploration →the model gets stuck in local optima (always giving the
same safe answer). Too much →training is unstable, quality fluctuates wildly.
Review: Chapters 3 and 7 (Introduction to RL; GRPO).
27.2
Core Algorithm Questions
Q1: Explain PPO’s clipped objective. Why does it work better than vanilla PG?
Answer: Vanilla policy gradient: ∇J = E[∇log π(a|s) · ˆA]. Problem: one lucky/unlucky sample
can produce a huge gradient →policy jumps to a bad region →generates garbage →next gradient
makes it worse →unrecoverable “death spiral.”
PPO’s solution: Clip the probability ratio r = πnew/πold to [0.8, 1.2].
Mechanics: For good actions ( ˆA > 0): objective is min(r ˆA, 1.2 ˆA). Once r exceeds 1.2, no further
benefit — stops the policy from over-committing to one example. For bad actions ( ˆA < 0):
objective is min(r ˆA, 0.8 ˆA). Once r drops below 0.8, penalty stops growing — prevents catastrophic
forgetting.
Key insight: It’s a first-order approximation of TRPO’s KL constraint, but without expensive
second-order optimization. Each update changes the policy by at most ±20%.
For LLMs specifically: The token-level ratio rt = πθ(yt|y<t)/πold(yt|y<t) prevents any single
token’s probability from changing too drastically, preserving coherent generation.
Review: Chapter 5 (PPO).
Q2: Derive DPO from first principles. What assumptions does it make?
Answer: Start with RLHF objective: maxπ E[r(x, y)] −βDKL[π∥πref].
Step
1:
Write the KKT conditions.
Optimal policy has closed form:
π∗(y|x)
∝
πref(y|x) exp(r(x, y)/β).
Step 2: Invert to express reward: r(x, y) = β log(π∗/πref) + β log Z(x).
Step 3: Substitute into Bradley-Terry model P(yw ≻yl) = σ(r(yw) −r(yl)). The partition
function Z(x) cancels (same prompt).
Step 4: Replace π∗with πθ (parameterized policy we’re training): L = −E[log σ(β log πθ(yw)
πref(yw) −
β log πθ(yl)
πref(yl))].
515


<!-- page 516 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Assumptions:
1. Bradley-Terry preference model (pairwise, no ties, transitive)
2. Optimal policy is achievable by πθ (sufficient capacity)
3. Preferences are generated from the same distribution as training data (no distribution shift)
4. Reference model is fixed and reasonable
When assumptions break: Real preferences aren’t transitive, data shifts during training, labels
are noisy →that’s why Online DPO and IPO exist.
Review: Chapter 6 (DPO).
Q3: GRPO vs PPO — when would you choose each? What’s the trade-off?
Answer:
GRPO advantages:
• No value function needed: saves one model’s worth of memory and complexity
• Simpler: fewer hyperparameters, more intuitive (above-mean = good, below-mean = bad)
• Better for verifiable rewards: math/code where r ∈{0, 1} gives crisp signal
• DeepSeek-R1 proved it can teach emergent reasoning with just binary rewards
PPO advantages:
• Per-token credit assignment: value function assigns reward to each token, not just sequence-
level
• More sample efficient: GAE uses value predictions to estimate advantage without generating
G samples
• Better for nuanced rewards: when reward is continuous and varies significantly across tokens
• More mature: battle-tested at OpenAI, Anthropic, etc.
Rule of thumb: If rewards are verifiable (right/wrong) →GRPO. If rewards are nuanced (RM
scores) and you need max quality →PPO. If compute is limited →GRPO (no critic training).
Compute comparison: GRPO generates G responses per prompt (8× more generation), but
skips value function training. Net: similar total compute but distributed differently (more gen,
less training).
Review: Chapters 5 and 7 (PPO; GRPO).
Q4: How does GAE work? Walk through a concrete example for LLMs.
Answer: GAE = weighted sum of n-step TD errors: ˆAt = PT−t
l=0 (γλ)lδt+l.
Concrete example: Response has 5 tokens. Reward only at end (r5 = 0.8). Value predictions:
V1 = 0.5, V2 = 0.55, V3 = 0.6, V4 = 0.65, V5 = 0.7.
TD errors (γ = 1): δ1 = 0 + V2 −V1 = 0.05, δ2 = 0 + V3 −V2 = 0.05, ..., δ5 = 0.8 + 0 −0.7 = 0.1.
With λ = 0.95:
ˆA5 = 0.1 (just the final TD error), ˆA4 = 0.05 + 0.95 × 0.1 = 0.145, ˆA3 =
0.05 + 0.95 × 0.145 = 0.188, etc.
Interpretation: Token 3 gets advantage 0.188 because it contributed to a sequence that got
higher reward than expected. Earlier tokens get credit through the exponential decay.
For LLMs: γ = 1.0 (all tokens matter, finite horizon). The advantage at token t answers: “given
what followed this token, was this token choice better or worse than expected?”
Review: Chapter 5 (PPO).
516


<!-- page 517 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q5: How do you prevent reward hacking? Give a layered defense strategy.
Answer: Detection signals: RM score rising but win-rate flat/declining. Response length
growing monotonically. KL divergence > 15. Diversity (unique n-grams) dropping. Reading
high-reward outputs reveals exploits.
Layered defense (in priority order):
1. KL penalty (primary): Adaptive controller targets KL ≈6. If KL rises, β increases
automatically. Prevents drifting too far from reference.
2. Reward model ensemble (3–5 models): Use min or mean of scores. Individual models
have different blind spots — exploits that fool one rarely fool all.
3. Length penalty: r′ = r −c · max(0, length −Ltarget). Prevents “just generate longer =
higher score” exploit.
4. Periodic RM refresh: Every 2000 steps, generate data from current policy, relabel, add
to RM training set. Closes the exploit as model finds it.
5. Win-rate based stopping: Track win-rate against SFT baseline. If RM score rises but
win-rate stalls for 200+ steps, stop training. The model is exploiting, not improving.
Post-detection recovery: Roll back to last “clean” checkpoint. Increase β by 2×. Add the
discovered exploit to the RM’s negative examples.
Review: Chapters 9 and 11 (Reward Model Training; System Architecture).
27.3
System Design Questions
Q6: Design an RLHF system for training a 70B model. Walk through every compo-
nent.
Answer: Three-cluster decoupled architecture on 72 A100-80GB GPUs:
Cluster 1 — Generation (32 GPUs):
• 8 vLLM instances, TP=4 each. PagedAttention + speculative decoding (1B draft model).
• Continuous batching, max 256 sequences in flight. INT8 weights for bandwidth.
• Output: (prompt, response, per-token log-probs). Throughput: ∼500 responses/minute.
• Stateless: just loads latest weights from shared store. Trivial recovery.
Cluster 2 — Scoring (8 GPUs):
• Reward model (70B, INT8 = 70GB) on 4 GPUs (TP=4).
• Reference model (70B, INT8) on 4 GPUs (TP=4). Computes per-token log-probs for KL.
• Output: reward scores + KL per token. Lightweight, batch inference.
Cluster 3 — Training (32 GPUs):
• Policy model with FSDP (ZeRO-3). Flash Attention 2. Gradient checkpointing.
• Consumes scored experiences from buffer. PPO update: 4 epochs on mini-batch of 16.
• Pushes updated weights to shared store every 50 steps (async, background transfer).
• Async checkpoint every 100 steps (non-blocking write to NVMe + S3 backup).
Connection fabric:
517


<!-- page 518 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Gen →Score: Ray/Redis queue (∼10 MB per batch: token IDs + log-probs)
• Score →Train: Experience buffer (circular, holds last 500 steps)
• Train →Gen: Weight store on shared parallel FS (Lustre/GPFS). 140GB push every 50
steps, async.
Overlap: While training processes step N, generation is already producing data for step N + 1.
This hides generation latency, achieving 1.3–1.5× throughput vs monolithic.
Review: Chapter 11 (System Architecture & Infrastructure at Scale).
Q7: How do you handle the generation bottleneck? Quantify the solutions.
Answer: Generation = 60–70% of total RLHF wall-clock time. Root cause: autoregressive
decoding is memory-bandwidth bound (arithmetic intensity ≈1 FLOP/byte vs roofline of 156
FLOP/byte on A100).
Solutions ranked by impact:
1. Decouple gen from training (1.3–1.5× end-to-end): Run generation on separate hardware,
overlap with training. The single biggest architectural win.
2. vLLM with PagedAttention (2–4×): Eliminates 60–80% KV cache memory waste from
internal fragmentation. Enables 3–4× larger batches = better bandwidth utilization.
3. Continuous batching (1.5–2×): Don’t wait for longest sequence to finish. Start new sequences
immediately in freed slots. Keeps GPUs busy.
4. Speculative decoding (2–3×): 1B draft model proposes 5 tokens. 70B model verifies all 5 in
one forward pass (parallel!). Accept 3–4 on average →3–4 tokens per forward pass instead of 1.
5. INT8/FP8 for generation weights (2×): Halves the 140GB weight read per token. Quality
loss is minimal because (a) we’re sampling with temperature anyway, and (b) only generation uses
INT8, training stays BF16.
6. CUDA graphs + kernel fusion (1.1–1.3×): Eliminate Python/CUDA launch overhead.
Fuse layernorm+attention+MLP into fewer kernels.
Combined: 1.5 × 3 × 1.5 × 2.5 × 2 × 1.2 = 40× over naive. In practice, diminishing returns limit
total to ∼10–20× over naive implementation.
Review: Chapters 2 and 11 (Systems Foundations; System Architecture).
Q8: Explain weight synchronization in a decoupled system. How much staleness can
you tolerate?
Answer:
The problem: Generation cluster uses policy weights to produce responses. Training cluster
updates those weights. They’re on different hardware. How to keep them in sync?
Why perfect sync is wasteful: Full sync of 70B BF16 = 140GB. At InfiniBand 400Gb/s
(50GB/s): 2.8s per sync. If you sync every step (every 50–90s), you spend 3–5% of time on weight
transfer. Acceptable, but unnecessary.
Staleness tolerance analysis:
• Per-step policy change: ∼0.1% (measured by mean param delta)
• 50 steps: ∼5% cumulative drift
• PPO clip range: handles up to 20% probability ratio deviation
• Empirical: 50-step staleness →<2% quality degradation (measured by win-rate)
Production strategy:
1. Every 50 training steps: push full BF16 checkpoint to shared store (2.8s transfer)
2. Generation cluster: non-blocking weight reload between batches
518


<!-- page 519 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
3. Delta compression (optional): Only send changed parameters (INT8 delta ≈5GB), apply as
offset. 10× less bandwidth.
4. For very large scale (256+ GPUs): streaming sync — continuously send small chunks in
background. Average staleness: 5–10 steps.
Important subtlety: Log-probs computed during generation use stale weights. The PPO ratio
πnew/πold computes πold using these stale log-probs. This is fine because PPO is designed for
off-policy corrections.
Review: Chapter 11 (System Architecture & Infrastructure at Scale).
Q9: How would you make this fault-tolerant at 512 GPUs?
Answer: At 512 GPUs, MTBF is 4–8 hours. A 5-day training run will see 15–30 failures.
Architecture-level resilience:
• Generation cluster = stateless. Failed instance restarts in <60s (just load weights, no state).
• Training cluster = stateful. Needs checkpoint-based recovery.
• Scoring cluster = stateless. Same as generation.
Checkpointing strategy:
• Frequency: Every 50–100 steps (5–10 min of training).
• Method: Async (non-blocking). Background thread writes while next step proceeds. Uses
FSDP’s distributed save (each rank saves its shard in parallel).
• Contents: Model weights, optimizer states (Adam m/v), LR scheduler, RNG states, KL
adaptive coefficient, global step counter, replay buffer pointer.
• Storage: Local NVMe (fast, 30s for 70B) + async copy to S3/shared FS (durable).
• Retention: Keep last 3 checkpoints. Auto-delete older ones.
Detection and recovery flow:
1. NCCL collective timeout (60s) or heartbeat miss (10s) →failure detected.
2. Identify failed node(s) via NVML health check.
3. Option A (fast, <2 min): Torch Elastic shrinks world size, redistributes shards, continue
with N −1 nodes. Request replacement in background.
4. Option B (clean, ∼5 min): Bring up replacement node, rebuild process group, load last
checkpoint, resume.
5. Experience buffer is persisted — no regeneration needed.
Prevention: Pre-screening stress test (GEMM, memory, NVLink).
ECC error monitoring
(preemptive migration if errors spike). Hot spare nodes (pre-loaded with environment). Dual-rail
InfiniBand for network redundancy.
Review: Chapter 11 (System Architecture & Infrastructure at Scale).
Q10: How do you scale from 7B to 70B to 405B? What changes at each scale?
Answer:
7B (single 8-GPU node, hours):
519


<!-- page 520 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Architecture: Monolithic (TRL default). All models on same GPUs.
• Memory: LoRA + INT8 ref/RM fits in 8×80GB easily.
• Parallelism: DP=8 or FSDP within node. No network communication.
• Hyperparams: LR=5 × 10−6, aggressive β=0.02, 50K–100K steps.
• Time: 4–12 hours per run. Fast iteration.
70B (32–64 GPUs, 2–5 days):
• Architecture: Semi-decoupled. vLLM generation + FSDP training.
• Memory: ZeRO-3 essential. Gradient checkpointing. INT8 ref/RM.
• Parallelism: TP=8 intra-node (gen), FSDP inter-node (training).
• Hyperparams: LR=1.5 × 10−6, moderate β=0.05, 10K–30K steps.
• Fault tolerance: Async checkpoints, monitoring, but manageable manually.
405B (256–512 GPUs, 1–3 weeks):
• Architecture: Fully decoupled. Separate clusters. Weight store + queue.
• Memory: ZeRO-3 + TP=8 + PP=2 for training. INT4 generation.
• Parallelism: 3D parallelism (TP×PP×DP = 8×2×16 = 256 GPUs training).
• Hyperparams: LR=5 × 10−7, very conservative β=0.1, 2K–5K steps.
• Fault tolerance: Mandatory. Elastic training, redundant checkpoints, hot spares.
• Key change: Much less RL training needed (model is already very capable from pretraining).
But each step is 50× more expensive, so instability is catastrophic.
Paradox: Larger models are actually easier to RL-train per step (more stable, smoother loss
landscape). But the cost of instability scales with model size — a bad run at 405B wastes $100K+
in compute.
Review: Chapter 11 (System Architecture & Infrastructure at Scale).
27.4
Practical and Debugging Questions
Q11: Reward score is increasing but model quality is declining. Diagnose and fix.
Answer: Classic reward hacking / Goodhart’s Law.
Diagnostic protocol:
1. Check response length: Plot mean length over training. Growing monotonically? =
Length exploit (RM gives higher scores to longer responses).
2. Check KL divergence: >15?
= Policy has diverged too far from reference.
Lost
capabilities.
3. Check diversity: Unique trigrams per response. Dropping? = Mode collapse (repeating
same high-reward pattern).
4. Manual inspection: Read 20 highest-reward responses. What pattern do they share? (e.g.,
all start with “Great question!”, all use bullet points, excessive hedging).
520


<!-- page 521 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
5. Win-rate: Evaluate against SFT baseline on held-out prompts. If flat/declining while RM
rises = confirmed exploit.
Immediate fixes:
• Roll back to last checkpoint where win-rate was improving
• Increase β by 2–3× (stronger KL penalty)
• Add explicit length penalty: r′ = r −0.001 · max(0, len −500)
Structural fixes (prevent recurrence):
• RM ensemble: Train 3–5 RMs on different data splits. Use min or mean. Exploits are
model-specific.
• RM refresh: Every 2000 steps, generate from current policy, get human labels, retrain RM.
• Multi-objective reward: Combine helpfulness + harmlessness + conciseness with separate
RMs.
• Early stopping on win-rate (not RM score). The metric you optimize should differ from
training signal.
Review: Chapters 9 and 11 (Reward Model Training; System Architecture).
Q12: How do you decide the prompt distribution for RL training?
Answer: Prompt quality is the most underrated factor in RLHF. Bad prompts = no learning
signal.
Composition (my default mix):
• 40% real user traffic (represents actual use cases)
• 30% synthetic (LLM-generated, fills gaps in coverage — rare topics, edge cases)
• 20% curriculum (graduated difficulty — start easy, increase complexity as model improves)
• 10% adversarial (red-team prompts, jailbreak attempts, ambiguous instructions)
Critical: The Goldilocks Filter:
1. For each candidate prompt, generate 4–8 responses with current model.
2. Score with RM. Compute pass rate (fraction above threshold).
3. Keep only prompts with 20–80% pass rate:
• <20%: Too hard. Model almost always fails →all negative advantages →no useful
gradient.
• >80%: Too easy. Model almost always succeeds →all positive advantages →no
contrast.
• 20–80%: Perfect. Mix of successes and failures →clear signal about what works.
4. Re-filter every 500 training steps (model improves, difficulty distribution shifts).
Topic diversity: Ensure no single topic dominates (<10% per category).
Use embedding
clustering to verify coverage. Otherwise the model over-optimizes for dominant topics.
Review: Chapters 7 and 9 (GRPO; Reward Model Training).
521


<!-- page 522 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q13: LoRA vs full fine-tuning for RLHF. When would you use each?
Answer:
LoRA (Low-Rank Adaptation, r=64, α=16):
• Trainable params: ∼0.2% of model (200M for 70B)
• Memory savings: No separate reference model needed (base model = reference)! Saves
140GB.
• Stability: Inherently more stable (low-rank constraint limits how far policy can drift)
• Speed: Faster per-step (fewer params to update), but may need more steps
• Quality ceiling: 90–95% of full FT quality typically
Full fine-tuning:
• All parameters updated. Maximum expressiveness.
• Needs separate reference model copy (140GB for 70B). Or very frequent checkpointing to
“anchor.”
• Higher risk of catastrophic forgetting. Need lower LR (3× lower than LoRA) and stronger β.
• Better when: large distributional shift needed (new language, very different style), LoRA
hits capacity limit.
My decision framework:
1. Start with LoRA (r=64). It’s 3× cheaper and more stable.
2. Monitor gradient norms on LoRA matrices. If consistently >1.0 (high relative to parameter
count): LoRA is capacity-limited.
3. Switch to full FT only when: win-rate plateaus AND gradient analysis suggests capacity
limitation.
4. For full FT: use LR/3, β × 2, more frequent checkpointing, and early stopping based on
win-rate.
Hybrid: LoRA for alignment/safety (small behavioral shift) + Full FT for capabilities/reasoning
(large shift needed).
Review: Chapters 1 and 10 (LLM Architecture; SFT Best Practices).
Q14: Process Reward Models (PRM) vs Outcome Reward Models (ORM). Design a
PRM system.
Answer:
ORM: Scores the final answer only. “Is the response good overall?” Simple but can’t identify
where reasoning went wrong.
PRM: Scores each intermediate step. “Is step 3 of this derivation correct?” Much more informative
but harder to train.
PRM advantages for reasoning:
• Identifies exactly where reasoning fails (step-level credit assignment)
• Enables tree search: expand only branches with high step scores
• Less reward hacking: can’t get high score from wrong steps + lucky final answer
522


<!-- page 523 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• PRM + best-of-N beats ORM + best-of-N by 10–20% on MATH benchmarks
Training a PRM:
1. Data collection (Monte Carlo approach):
• For each problem, generate reasoning trace step by step
• At each step k, complete the solution M times (M = 32) from that point
• Step score = fraction of completions that reach correct answer
• Steps where score drops significantly = the “mistake” steps
2. Labeling: Step is “correct” if its completion rate > 50%, “incorrect” if < 20%.
3. Model: Same architecture as base model + classification head per token position. Train
with binary cross-entropy on step labels.
4. Inference: Score each step. If any step scores <0.3, flag the trace as flawed.
Using PRM in RLHF: Per-token rewards from PRM feed directly into GAE. Each token gets
immediate feedback, not just end-of-sequence. This dramatically improves credit assignment for
long reasoning chains.
Review: Chapters 9 and 13 (Reward Model Training; RL for Large Reasoning Models).
Q15: How do you evaluate whether RL actually improved the model?
Answer: Multi-faceted evaluation (no single metric captures “quality”):
1. Win-rate (most important, most reliable):
• 500+ diverse prompts. LLM judge (GPT-4 or Claude) picks winner in blind A/B comparison
vs SFT baseline.
• Target: >55% win-rate = meaningful improvement. >65% = strong improvement.
• Use position-debiasing (swap A/B order, average). Report confidence intervals.
2. Capability benchmarks (regression detection):
• MMLU (knowledge), HumanEval (code), MATH (reasoning), MT-Bench (multi-turn).
• Any >2% drop = concerning alignment tax. Investigate which categories degraded.
3. Category-specific evals:
• Safety: refusal rate on harmful prompts (should increase)
• Truthfulness: TruthfulQA score (should increase or stay flat)
• Helpfulness: task completion rate on instruction-following benchmarks
4. Distributional metrics:
• Response length distribution (shouldn’t shift dramatically)
• Vocabulary diversity (unique tokens per response)
• Format compliance (if trained for specific format)
5. Human evaluation (gold standard, expensive):
523


<!-- page 524 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Blind A/B with 3+ skilled annotators per example. Inter-annotator agreement > 70%.
• Only for final model selection, not during training (too slow/expensive).
Red flags: RM score up + win-rate flat = reward hacking, not real improvement. Win-rate up +
benchmarks down = alignment tax too high, reduce RL strength.
Review: Chapter 14 (LLM Evaluation).
Q16: Describe the reward model training pipeline end-to-end.
Answer:
Phase 1 — Data Generation:
• Collect 50K–100K diverse prompts (real traffic + synthetic)
• Generate 4–8 responses per prompt at varying temperatures (0.3, 0.7, 1.0) and from multiple
models (diversity is key — if all responses are similar, preferences are uninformative)
• Total: 200K–800K candidate responses
Phase 2 — Preference Collection:
• Option A (expensive, best quality): Human annotators compare pairs. 3 annotators per
pair. Cost: $2–5 per comparison.
• Option B (cheap, 85–90% agreement with humans): LLM judge (GPT-4/Claude). 10×
cheaper. Good for scale.
• Format: (prompt, chosen response, rejected response). Pairs with annotator disagreement
(<70% agreement) are discarded.
• Final dataset: 100K–500K pairs.
Phase 3 — Training:
• Architecture: Same as base LLM + scalar head (one regression output per sequence).
• Loss: L = −E[log σ(r(x, yw) −r(x, yl))] (Bradley-Terry).
• Training: 1 epoch only! RMs overfit extremely fast. Validation accuracy 68–75% is good
(higher often means overfitting to annotation artifacts).
• Tricks: Center rewards around 0 (subtract running mean). Check for length bias (if correlation
between length and score > 0.3, add length penalty to training).
Phase 4 — Validation:
• Hold-out preference pairs: accuracy should be 68–75%.
• Agreement with humans on new data: > 80%.
• Length bias check: correlation between response length and RM score should be < 0.2.
• Consistency check: same prompt, paraphrased responses should get similar scores.
Review: Chapter 9 (Reward Model Training).
524


<!-- page 525 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q17: What happens when KL divergence explodes? Root cause and fix.
Answer:
What KL measures:
Average log-ratio between current policy and reference:
DKL =
Ey∼πθ[log(πθ(y|x)/πref(y|x))].
KL=0 means identical to reference.
KL=10 means the policy
puts 10 nats more probability on its preferred outputs.
Healthy range: 3–10 during training. Slowly increasing is fine. Sudden spike = problem.
Root causes of KL explosion:
1. Learning rate too high: Policy takes giant step, diverges from reference. Fix: reduce LR
by 2–5×.
2. Reward hacking: Found an exploit that gets high reward far from reference behavior. Fix:
increase β, add RM ensemble.
3. Mode collapse: Policy concentrates on one response template. KL is high at that template,
low everywhere else. Fix: increase entropy bonus, increase temperature.
4. Bad batch: One unlucky batch with extreme advantages pushed the policy. Fix: gradient
clipping, reduce mini-batch size.
5. Value function diverged: Wrong advantage estimates cause wrong updates. Fix: reduce
value function LR, or switch to GRPO (no value function).
Recovery protocol:
1. Detect: KL > 15 for 50+ steps, or KL jumps >5 in one step.
2. Immediate: Load last clean checkpoint (KL < 10).
3. Adjust: Reduce LR by 50%. Increase β by 2×. Lower cliprange to 0.1.
4. Resume: Monitor closely for first 200 steps.
Review: Chapters 5 and 7 (PPO; GRPO).
Q18: Compare monolithic vs decoupled RLHF architectures. When does each make
sense?
Answer:
Monolithic (TRL default: single process, all models on same GPUs):
• Pros: Simple code. No distributed systems complexity. Easy debugging.
• Cons: GPUs idle 60% of time (compute idle during gen, bandwidth idle during training).
Doesn’t scale past ∼16 GPUs efficiently. All models compete for same memory.
• Use when: Model ≤13B, single node, research/prototyping.
Semi-decoupled (vLLM gen + FSDP training, same cluster):
• Pros: Better utilization (gen and training can partially overlap). Scales to 64 GPUs.
• Cons: Still shares hardware, can’t optimize independently. More complex than monolithic.
• Use when: 13B–70B, 2–8 nodes, production experiments.
Fully decoupled (separate clusters connected by queues):
• Pros: Each cluster optimized for its workload. Gen scales independently from training. Gen
cluster is stateless (trivial fault tolerance). Scales to hundreds of GPUs.
525


<!-- page 526 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Cons: Distributed systems complexity. Weight staleness. Queue management. Network
overhead.
• Use when: ≥70B production training. Need scale, fault tolerance, high utilization.
The key insight: Generation is bandwidth-bound, training is compute-bound. Same hardware
can’t optimize for both. Decoupling lets gen nodes have: more memory bandwidth, INT8 weights,
large batch. Training nodes have: full BF16 precision, Flash Attention, FSDP sharding.
Review: Chapter 11 (System Architecture & Infrastructure at Scale).
Q19: How do you set up curriculum learning for RL training?
Answer: Curriculum = gradually increasing difficulty so the model learns progressively.
Why it matters: If you throw the hardest prompts at a weak model, it gets all-negative rewards
→no learning signal (everything is equally bad). If you start easy, the model develops basic
capabilities, then builds on them.
Implementation:
1. Difficulty scoring: Rate each prompt by current model’s pass rate (from Goldilocks
filtering). Easy = >80% pass, Medium = 30–80%, Hard = <30%.
2. Schedule: Steps 0–1000: 70% easy, 20% medium, 10% hard. Steps 1000–5000: 30% easy,
50% medium, 20% hard. Steps 5000+: 10% easy, 40% medium, 50% hard.
3. Dynamic adjustment: Every 500 steps, re-evaluate difficulty distribution. Prompts the
model has “mastered” (pass rate > 95%) get retired. New harder prompts are introduced.
4. For GRPO specifically: Curriculum ensures groups always have a mix of successes and
failures. Without curriculum, hard prompts give all-zero groups (useless).
Evidence: DeepSeek-R1 used implicit curriculum — starting with easy math/code problems,
the model developed basic reasoning, then solved progressively harder problems without explicit
scheduling.
Review: Chapters 7 and 12 (GRPO; LLM Agentic Training).
Q20: You have a budget of 64 A100-80GB GPUs and need to RL-train a 70B model.
Design the allocation.
Answer: 8 nodes × 8 GPUs = 64 total. Need to split across generation, scoring, and training.
My allocation:
• Generation: 24 GPUs (3 nodes).
6 vLLM instances with TP=4.
INT8 weights =
70GB/model, leaves room for KV cache. Continuous batching, batch ≈128 total.
• Scoring: 8 GPUs (1 node). RM (INT8, TP=4) + Reference model (INT8, TP=4) on same
node. Or share 4 GPUs with TP=4 alternating between RM and ref.
• Training: 32 GPUs (4 nodes). FSDP across all 32. Each GPU holds ∼70B/32 = 2.2GB
params + optimizer fraction. Plenty of headroom for activations. Gradient checkpointing
for safety.
Expected throughput:
• Generation: 6 instances × ∼80 responses/min = 480 responses/min
• Training: Batch of 128, one step every ∼15s (training only, no gen wait)
• Overlap: While training step N (15s), generation produces ∼120 responses for step N + 1.
Perfect pipeline.
526

