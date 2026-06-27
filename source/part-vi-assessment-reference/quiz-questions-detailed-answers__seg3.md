<!-- page 540 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
3. Rescale output: Onew = rescaled(Oold) + new contribution
This is mathematically exact — no approximation. It enables block-by-block processing where
each block fits in SRAM, never needing the full n × n matrix in memory.
Review: Chapters 1 and 2 (LLM Architecture; Systems Foundations).
27.13
LoRA and PEFT Questions
Q: Why does LoRA work? What theoretical insight justifies low-rank updates?
Answer: Aghajanyan et al. [93] showed that fine-tuning operates in a very low intrinsic dimen-
sionality — the effective parameter space for a fine-tuning task is far smaller than the model’s
total parameter count. A 175B model may have intrinsic dimensionality <10,000 for a given task.
LoRA directly exploits this: by constraining updates to rank r (W ′ = W + BA, B ∈Rd×r), it
restricts learning to an r-dimensional subspace per weight matrix. Since the true task subspace is
low-dimensional, this loses almost nothing while reducing trainable parameters by 100–1000×.
Intuition: Fine-tuning doesn’t change what the model “knows” (the full-rank W stays frozen); it
only adjusts how existing knowledge is combined for the new task — a low-rank perturbation.
Review: Chapter 1 (LLM Architecture and Optimization Methods).
Q: Compare QLoRA vs. full LoRA vs. full fine-tuning for a 70B model. When would
you choose each?
Answer:
Method
Memory
GPUs
Quality
Use When
Full fine-tune
560+ GB
8+ A100
Best
Unlimited
budget;
pre-
training continuation
LoRA (r = 16)
145 GB
2 A100
95–98%
Good budget; general fine-
tuning
QLoRA (r = 16)
44 GB
1×48GB
93–96%
Single-GPU; prototyping; con-
strained resources
Decision tree: (1) If task requires deep knowledge change →full fine-tune. (2) If adapting to
new style/format →LoRA. (3) If memory-constrained or rapid iteration →QLoRA. (4) If rank
matters: start r = 16; increase if training loss plateaus above full fine-tune level.
Review: Chapters 1 and 10 (LLM Architecture; SFT Best Practices).
Q: What is DoRA and why does it outperform standard LoRA?
Answer: DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes W into magnitude
∥W∥and direction W/∥W∥, then applies LoRA only to the direction component:
W ′ = m ⊙
W + BA
∥W + BA∥
where m (magnitude) is also trainable but as a simple scalar per output neuron.
Why it helps: Full fine-tuning naturally updates both magnitude and direction independently.
Standard LoRA couples them (the low-rank update changes both simultaneously in a constrained
way). DoRA decouples them, giving LoRA the same “degrees of freedom” structure as full
fine-tuning. Result: 1–3% improvement on reasoning tasks with no extra compute at inference
(merge adapters).
Review: Chapter 1 (LLM Architecture and Optimization Methods).
540


<!-- page 541 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
27.14
Model Compression Questions
Q: Explain AWQ. Why does protecting 1% of weights preserve 99% of quality?
Answer: AWQ (Activation-Aware Weight Quantization) observes that weight importance is highly
non-uniform: weights that multiply large activations contribute disproportionately to the output.
The key insight: ∥W · X∥depends on both W and X. A small weight multiplied by a large
activation matters more than a large weight multiplied by a near-zero activation.
AWQ identifies the top 1% of “salient” channels (those with consistently large activation magnitudes
across calibration data) and protects them by scaling: multiply salient channels by a factor s > 1
before quantization (then divide by s in the activation). This reduces relative quantization error
for important channels.
Result: 4-bit quantization with <1% quality loss on 70B models, because the 99% of non-salient
weights can tolerate aggressive quantization.
Review: Chapters 1 and 2 (LLM Architecture; Systems Foundations).
Q: When should you use FP8 vs. 4-bit quantization vs. BF16?
Answer:
• BF16: Training (policy model in RLHF), when precision matters. Default for any model
being updated by gradients.
• FP8 (E4M3): H100 training with Transformer Engine (2× throughput, <0.5% quality
loss). Also for inference on H100 when you need maximum throughput.
• INT8/FP8 inference: Frozen models in RLHF (reference model, reward model) — not
being trained, so reduced precision is safe.
• 4-bit (AWQ/GPTQ): Inference serving at scale.
Best memory/quality tradeoff for
deployment. Also for QLoRA base model.
• 2-bit: Experimental; edge deployment where memory is extreme constraint. Quality loss
5–10%.
Rule: quantize as aggressively as possible for inference, keep BF16 (or FP8 on H100) for training.
Review: Chapter 2 (Systems Foundations for LLMs).
Q: Explain NVIDIA 2:4 structured sparsity. What’s the speedup and constraint?
Answer: 2:4 sparsity means: in every group of 4 consecutive elements, exactly 2 must be zero.
This is enforced at the weight level.
Hardware support: A100/H100 Tensor Cores have dedicated 2:4 sparse GEMM instructions
that skip the zero elements, achieving exactly 2× throughput with no software overhead.
Constraint: You must achieve exactly 50% sparsity in this specific pattern. You can’t have
30% or 70% sparsity; you can’t have arbitrary sparsity patterns. The pruning must respect the
4-element group structure.
How to achieve it: After training (or during fine-tuning), for each group of 4 weights, zero out
the 2 smallest by magnitude. Then fine-tune for a few hundred steps to recover quality. Quality
loss: typically <1% for large models (70B+).
Review: Chapter 2 (Systems Foundations for LLMs).
541


<!-- page 542 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
27.15
Mixture of Experts Questions
Q: Mixtral 8x7B has 47B total parameters but only 13B active per token. Explain
how this works and why it’s efficient.
Answer: Mixtral replaces each FFN layer with 8 parallel expert FFNs (each ∼7B params for the
FFN portion). A router network selects the Top-2 experts per token.
Why 47B total: Attention layers are shared (not replicated) = ∼5B. FFN experts: 8 × ∼5.25B
= 42B. Total: ∼47B.
Why 13B active: Per token, only 2 experts fire. Active params = attention (∼5B) + 2 FFN
experts (∼2 × 5.25B) ≈13B.
Why efficient: Compute cost scales with active params (13B), matching a 13B dense model. But
capacity (knowledge stored) scales with total params (47B), matching much larger models. Result:
Mixtral matches Llama-2 70B quality at 13B compute cost.
Memory cost: Still need all 47B params in memory (all experts loaded), so memory = 47B
model, but compute = 13B model.
Review: Chapter 1 (LLM Architecture and Optimization Methods).
Q: What is the load balancing problem in MoE and how is it solved?
Answer: Without constraints, the router tends to send most tokens to 1–2 “popular” experts
(rich-get-richer dynamics). This causes:
• Capacity waste: 6 of 8 experts are unused, model effectively shrinks to 2-expert size.
• Compute imbalance: If each expert is on a different GPU, popular experts become bottlenecks
while others idle.
Solution: Auxiliary load-balancing loss: Lbal = α · N PN
i=1 fi · pi, where fi = fraction of tokens
routed to expert i, pi = mean router probability for expert i. This penalizes uneven distributions.
Alternative: Expert capacity factor — hard cap on max tokens per expert per batch. Overflow
tokens are dropped or re-routed.
Typical α: 0.01–0.1 (small enough not to hurt main loss, large enough to prevent collapse).
Review: Chapter 1 (LLM Architecture and Optimization Methods).
27.16
Diversity in Training Questions
Q: What happens if all N responses in a GRPO group are identical?
Answer: If all N responses are identical: all rewards ri are equal, so σG = 0 and advantages
ˆAi = (ri −µG)/σG are undefined (division by zero). In practice, implementations set ˆAi = 0 for
all, meaning zero learning signal — the step is wasted.
Prevention:
1. Temperature: Use τ = 0.7–1.0 during generation (not greedy).
2. Large N: N = 8–16 increases probability of diverse responses.
3. Duplicate rejection: DAPO’s approach — reject duplicate responses and resample.
4. Frequency penalty: Penalize repeated n-grams during generation.
5. Monitor: Track unique-response ratio per group. If <50%, increase temperature.
Review: Chapter 7 (GRPO).
542


<!-- page 543 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q: Explain the diversity-quality tradeoff in RLHF. How do you detect mode collapse?
Answer:
Tradeoff:
High diversity (high entropy/temperature) = varied but potentially
random/low-quality responses. Low diversity = consistent but repetitive, reward-hacked responses.
Detecting mode collapse (all should be monitored during training):
1. Response entropy: Compute per-token entropy H = −P pi log pi. If dropping rapidly →
collapse.
2. Unique n-gram ratio: Fraction of unique 4-grams across responses to same prompt.
Healthy: >0.6.
3. Reward distribution width: If σ(rewards) shrinks to near-zero →all responses are the
same quality →likely identical.
4. KL divergence: If DKL[πθ∥πref] is growing rapidly, the policy is moving far from reference
→often toward a narrow mode.
5. Length histogram: If all responses converge to same length →template behavior.
Fix: Increase KL coefficient β, increase entropy bonus, increase sampling temperature, or rollback
to earlier checkpoint.
Review: Chapters 7 and 9 (GRPO; Reward Model Training).
27.17
Speculative Decoding Questions
Q: Speculative decoding claims “no quality loss.” How can generating tokens differently
produce identical output distribution?
Answer: The acceptance/rejection scheme guarantees distributional equivalence:
For each draft token ˆx with draft probability q(ˆx) and target probability p(ˆx):
• Accept with probability min(1, p(ˆx)/q(ˆx))
• On rejection: sample from the residual distribution ∝max(0, p(x) −q(x))
This is mathematically equivalent to sampling directly from p (the target). Proof sketch: the
probability of outputting token x is q(x) · min(1, p(x)/q(x)) + P(reject) ·
max(0,p(x)−q(x))
P
y max(0,p(y)−q(y)) = p(x).
The speedup comes from amortizing: when the draft is good (high acceptance), multiple tokens
are confirmed in one target forward pass. The guarantee holds regardless of draft quality — bad
drafts just give lower speedup (more rejections), not worse quality.
Review: Chapter 2 (Systems Foundations for LLMs).
Q: Compare Medusa vs. Eagle for speculative decoding. When would you choose
each?
Answer:
Medusa: Adds k parallel prediction heads to the target model. Each head independently predicts
token at position t + i. Pro: No separate model, <1% memory overhead. Con: Heads predict
independently — cannot condition position t + 2 on what was predicted at t + 1. Acceptance rate:
60–80%.
Eagle: Lightweight autoregressive decoder on target model’s hidden states. Draft tokens are
generated autoregressively (each conditioned on previous). Pro: Captures inter-token dependencies
→85–95% acceptance rate. Con: Slightly more memory (small decoder) and sequential draft
generation.
Choose Medusa when: Memory is extremely tight; simple integration; moderate speedup is
sufficient (2–2.5×).
543


<!-- page 544 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Choose Eagle when: Maximum speedup needed (3–4×); can afford small extra model; latency-
critical single-stream generation.
Choose N-gram when: Repetitive outputs (code, structured data); zero cost; no training
needed.
Review: Chapter 2 (Systems Foundations for LLMs).
Q: Why does speculative decoding NOT help at high batch sizes?
Answer: At high batch sizes (≥64), autoregressive generation is already compute-efficient: the
weight-read cost is amortized across many sequences. The arithmetic intensity approaches the
roofline ridge point.
Speculative decoding adds overhead:
1. Draft generation cost (even if small model, it’s not free at high batch)
2. Verification forward pass processes k extra tokens per sequence (batch × k tokens)
3. Memory for draft model or Medusa heads
4. Rejected tokens waste compute
At batch=1 (latency-bound, memory-bound): speculation turns 1 token/step into 3–4 tokens/step
— huge win.
At batch=128 (already compute-efficient): the extra tokens from speculation barely help throughput
because the GPU is already near-saturated. The overhead may even reduce throughput.
Rule: Speculative decoding is for latency (small batch); batching is for throughput (large batch).
Don’t combine them.
Review: Chapter 2 (Systems Foundations for LLMs).
27.18
Agentic RL Questions
Q: Why does standard RLHF (single-turn PPO/DPO) fail for multi-step agents?
Answer: Standard RLHF optimizes for single-turn quality: given a prompt, produce one good
response. Multi-step agents face fundamentally different challenges:
1. Credit assignment: In a 50-step trajectory, which step caused the failure? Single-turn
reward assigns credit to the entire response uniformly; multi-step needs per-step credit.
2. Sparse rewards: Success/failure only at trajectory end. PPO’s GAE assumes intermediate
rewards; without them, advantage estimates are noisy.
3. Action space: Actions are structured tool calls (JSON), not just token sequences. The
model must learn syntax + semantics + strategy simultaneously.
4. Non-stationarity: The environment changes with each action (tool outputs modify state).
Each step has a different “prompt” unlike single-turn where input is fixed.
5. Exploration: Agent must discover novel tool-use strategies, not just rephrase text.
Solution: Trajectory-level GRPO (rank complete trajectories), process reward models (per-step
feedback), or filtered SFT on successful trajectories.
Review: Chapter 12 (LLM Agentic Training).
544


<!-- page 545 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q: Explain how GRPO is adapted for agentic training. What are the key differences
from single-turn GRPO?
Answer: Single-turn GRPO: generate N responses to a prompt, rank by reward, compute
advantages.
Agentic GRPO differences:
1. Unit of generation: A full trajectory (10–100 steps) instead of a single response. Each
trajectory is one “sample” in the group.
2. Reward: Terminal (task success/failure) or trajectory-level (sum of step rewards). NOT
per-token.
3. Masking: Only compute policy loss on the agent’s outputs (reasoning + tool calls). Mask
tool outputs (environment responses) from gradient computation.
4. Group size: Typically smaller (N = 4–8) because trajectories are expensive (many forward
passes per trajectory).
5. KL penalty: Applied per-step to prevent drift from SFT policy at each decision point.
6. Length normalization: Normalize by number of agent actions (not tokens) to avoid
penalizing thorough reasoning.
Review: Chapters 7 and 12 (GRPO; LLM Agentic Training).
Q: Compare STaR and Reflexion and ReAct for agents.
Answer:
STaR (Self-Taught Reasoner): Generate reasoning chains →filter by correctness →fine-tune
on correct ones. Use when: You have verifiable tasks (math, code) and want to bootstrap reasoning
from a base model without RL.
Reflexion: After failure, generate verbal feedback (“What went wrong?”) →retry with reflection
in context. No weight updates. Use when: Inference-time improvement; limited compute for
training; tasks where self-diagnosis is possible.
ReAct: Interleave Reasoning (think) + Acting (tool use) in a structured loop. Use when: Multi-
step tool use tasks; you need transparency (reasoning traces are interpretable); the agent must
decide between thinking and acting.
Key differences:
STaR
Reflexion
ReAct
Updates weights?
Yes (SFT)
No (in-context)
No (prompting)
Multi-step?
No (single reasoning chain)
Yes (retry loops)
Yes (think-act cycles)
Tools?
No
Optional
Yes (required)
Best for
Reasoning improvement
Error recovery
Tool-augmented tasks
Review: Chapters 12 and 18 (LLM Agentic Training; Agent Design Patterns).
Q: Why is GRPO preferred over PPO for a research agent?
Answer: For a research agent with 20–100 step trajectories:
PPO requires a value model: V (st) must predict expected total reward from the current state.
For research (where state = 128K tokens of context including papers, code, and results), training
an accurate value function is extremely difficult — the value of “having read 3 papers and written
partial code” is hard to predict.
GRPO avoids value estimation entirely: It generates N complete trajectories per research
question and uses within-group ranking as the advantage. No need to predict intermediate value —
just compare outcomes.
Additional reasons:
545


<!-- page 546 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Research quality is binary-ish (good report vs. bad report) — ranking is natural.
• Trajectories are long and expensive; GRPO’s N = 4 is manageable; PPO would need many
rollouts for stable value estimates.
• Terminal reward is sparse; GAE with sparse rewards gives noisy per-step advantages anyway.
Review: Chapters 7 and 12 (GRPO; LLM Agentic Training).
Q: Design a reward function for a coding agent. What reward hacking risks exist?
Answer: Reward design:
R = 0.5 · Rtests + 0.2 · Rquality + 0.2 · Refficiency + 0.1 · Rsafety
• Rtests: Fraction of unit tests passing (0–1). Ground-truth verifiable.
• Rquality: LLM judge on code style, documentation, maintainability.
• Refficiency: max(0, 1 −steps/30) — bonus for finishing quickly.
• Rsafety: No dangerous operations (rm -rf, network access outside sandbox).
Reward hacking risks:
1. Hardcoded outputs: Agent learns to print expected test outputs directly without comput-
ing them. Fix: Randomize test inputs; test on held-out cases.
2. Test deletion: Agent modifies/deletes failing tests. Fix: Sandbox tests as read-only.
3. Trivial solutions: Agent writes minimal code that passes tests but doesn’t generalize. Fix:
Large, diverse test suites; property-based testing.
4. Efficiency gaming: Agent skips reasoning steps to maximize efficiency bonus.
Fix:
Minimum quality threshold before efficiency bonus applies.
Review: Chapters 9, 12, and 19 (Reward Model Training; LLM Agentic Training; Agentic
Environments).
27.19
Listwise Rewards and Advanced RM Questions
Q: Explain the Plackett-Luce model. How does it generalize Bradley-Terry?
Answer: Bradley-Terry models pairwise preferences: P(y1 ≻y2) = σ(r(y1) −r(y2)).
Plackett-Luce models full rankings of K items as sequential selection:
P(π) =
K
Y
i=1
er(yπ(i))
PK
j=i er(yπ(j))
Interpretation: Sequentially pick the best remaining item. Position 1 = softmax over all K;
position 2 = softmax over remaining K −1; etc.
Generalization: For K = 2, PL reduces exactly to BT: P(y1 ≻y2) =
er(y1)
er(y1)+er(y2) = σ(r(y1) −
r(y2)).
Advantage: A ranking of K = 8 items provides
 8
2
 = 28 implicit pairwise comparisons plus
relative margin information — much richer training signal than a single pair.
Review: Chapter 9 (Reward Model Training).
546


<!-- page 547 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q: What is a Process Reward Model (PRM) and when is it better than an Outcome
Reward Model (ORM)?
Answer: ORM: Scores the final output only. r(x, yfinal) = one scalar for the complete response.
PRM: Scores each step of reasoning. r(x, ystep t) = scalar per intermediate step.
PRM is better when:
1. Long reasoning chains: Math problems with 10+ steps. ORM can’t tell which step went
wrong; PRM provides per-step credit assignment.
2. Search/verification: PRM enables tree search (beam search over reasoning steps, prune
branches with low step-reward).
3. Training signal density: PRM gives T rewards per trajectory (one per step) vs. ORM’s
single reward →lower variance advantage estimates.
ORM is better when: Tasks are short (single-turn); step boundaries are unclear; annotation
cost for per-step labels is prohibitive.
PRM annotation: Can be automated via “Math-Shepherd” approach: for each step, complete
the solution from that point multiple times. If completions from step t succeed but completions
from step t + 1 fail, step t + 1 is likely wrong.
Review: Chapters 9 and 13 (Reward Model Training; RL for Large Reasoning Models).
27.20
RL for Large Reasoning Models Questions
Q: Why does DeepSeek-R1 not use a Process Reward Model despite training on long
reasoning chains?
Answer: DeepSeek-R1 uses only outcome-based rewards (accuracy + format) for several
reasons:
1. Verifiable tasks: Math and code have deterministic ground-truth answers. The binary
accuracy reward provides sufficient signal even for long chains.
2. PRM failure modes: Step-level reward models introduce their own reward hacking — the
model can learn to produce steps that “look correct” to the PRM without actually being
correct.
3. GRPO’s group normalization: By sampling G completions per prompt and normalizing
advantages within each group, GRPO naturally provides relative signal about which reasoning
strategies work, even without per-step rewards.
4. Emergent self-correction: With outcome-only rewards, the model learns to self-correct
within chains (the “aha moment”), which wouldn’t emerge if per-step rewards micromanage
the reasoning process.
Key insight: The verifiability of the task domain is what makes PRMs unnecessary — for
subjective tasks (creative writing), outcome-only rewards may not suffice.
Review: Chapter 13 (RL for Large Reasoning Models).
Q: Explain the test-time compute scaling law and its implications for model deployment
Answer: The test-time compute scaling law states:
Accuracy(Ctrain, Ctest) ≈f(α log Ctrain + β log Ctest)
Implications:
1. Compute equivalence: A 7B model with 64× more inference tokens can match a 70B
547


<!-- page 548 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
model with 1× tokens on reasoning tasks.
2. Adaptive allocation: Easy questions get short chains (cheap); hard questions get long
chains (expensive). Average cost is lower than always using a large model.
3. Deployment flexibility: Instead of one large model, deploy a smaller reasoning model and
scale inference compute per-query based on difficulty.
4. Diminishing returns: The log relationship means doubling test-time compute gives
diminishing accuracy gains — there’s an optimal allocation between training and inference
compute.
The “overthinking” failure: Very long chains can decrease accuracy due to error accumulation
and attention dilution. Optimal chain length depends on problem difficulty.
Review: Chapter 13 (RL for Large Reasoning Models).
Q: How does MCTS (Monte Carlo Tree Search) apply to LLM reasoning?
Answer: MCTS for reasoning treats each partial solution as a tree node:
Four phases per iteration:
1. Selection: Navigate from root using UCB: UCB(s) = Q(s) + c
r
ln N(parent)
N(s)
2. Expansion: Generate new reasoning steps (child nodes) from the LLM
3. Simulation: Complete the solution from the new node (rollout)
4. Backpropagation: Update Q-values along the path based on final correctness
Key differences from game MCTS:
• Branching factor: Reasoning has enormous branching factor (any next sentence is possible).
Practical implementations use the LLM’s top-k outputs to limit branches.
• Value function: A trained PRM estimates partial solution quality, replacing random
rollouts.
• Step granularity: Each “step” might be one sentence, one equation, or one logical inference
— choosing granularity matters.
Used by: AlphaProof (math olympiad), and hypothesized for OpenAI o1/o3’s hidden reasoning.
Review: Chapter 13 (RL for Large Reasoning Models).
Q: Compare distillation vs direct RL for creating small reasoning models
Answer:
Distillation (DeepSeek-R1-Distill approach):
• Generate reasoning chains from large model (R1-671B)
• SFT small model on these chains
• Result: Small model mimics large model’s reasoning format
• Pro: Cheap (just SFT). Con: May learn surface patterns not true reasoning.
Direct RL on small model:
• Train small model with GRPO/PPO against verifiable rewards
548


<!-- page 549 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Model discovers its own reasoning strategies
• Pro: Genuine capability. Con: Much more compute; may not converge for very small models.
Empirical finding: R1-Distill-7B (distilled) outperforms direct-RL-7B on most benchmarks.
The reasoning chains from the large model provide such strong supervision that SFT alone is
competitive. However, distilled models show less generalization to truly novel problem types.
Best practice: Distill first (cheap baseline), then optionally run RL on the distilled model for
further gains (“distill + RL” combo used by Qwen).
Review: Chapter 13 (RL for Large Reasoning Models).
27.21
LLM Evaluation Questions
Q: Derive the ELO rating update rule and explain why Chatbot Arena uses it
Answer: ELO derivation:
Expected score of player A vs B: EA =
1
1+10(RB−RA)/400 (logistic model).
After a match with actual score SA ∈{0, 0.5, 1}: R′
A = RA + K(SA −EA)
The K-factor controls update magnitude (higher K = more reactive to recent results).
Why Chatbot Arena uses ELO:
1. Transitivity: If model A beats B and B beats C, ELO predicts A beats C. This gives a
total ordering from pairwise comparisons.
2. Online updates: New models can be added without re-evaluating all pairs. Each new
comparison updates ratings incrementally.
3. Confidence: After N comparisons, rating uncertainty shrinks as O(1/
√
N). Standard error:
SE ≈400
√
N .
4. Human preference capture: Real users provide honest preferences without needing to
articulate criteria. The aggregate reveals true model quality.
Chatbot Arena specifics: Uses Bradley-Terry MLE (equivalent to ELO at convergence) with
bootstrap confidence intervals. Style-controlled ELO removes length/formatting bias.
Review: Chapter 14 (LLM Evaluation).
Q: What is the pass@k metric for code generation and why is the unbiased estimator
important?
Answer: pass@k = probability that at least one of k generated samples passes all test cases.
Naive (biased) estimator: Generate k samples, check if any passes. Problem: high variance,
expensive (need many trials per problem).
Unbiased estimator (Chen et al., 2021): Generate n ≥k samples, count c that pass:
pass@k = 1 −
 n−c
k

 n
k

Why unbiased matters:
1. Generates n samples once (e.g., n = 200) and computes pass@1, pass@10, pass@100 from
the same samples
2. No need to repeat the entire evaluation k times
3. Statistically exact (combinatorial argument: fraction of k-subsets with no correct sample)
4. Numerically
stable
computation
via
log-space:
pass@k
=
1
−
exp
Pk−1
i=0 log(n −c −i) −log(n −i)

549


<!-- page 550 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Intuition: If 50/200 samples pass (c = 50, n = 200), pass@1 ≈0.25, pass@10 ≈0.94. The
estimator counts what fraction of k-sized draws would contain at least one success.
Review: Chapters 14 and 19 (LLM Evaluation; Agentic Environments).
Q: How do you detect and mitigate benchmark contamination?
Answer: Contamination: Training data contains benchmark test examples (or close para-
phrases), inflating scores.
Detection methods:
1. N-gram overlap: Check if training data contains exact or near-exact matches to test items.
8-gram overlap with >80% coverage is suspicious.
2. Canary strings: Insert unique identifiers in test sets; check if model can reproduce them.
3. Rephrased benchmarks: Create semantically equivalent but textually different versions
of benchmarks. Large accuracy drops suggest memorization.
4. Temporal analysis: Model performance on pre-training-cutoff vs post-cutoff test items.
Unusually high performance on old items suggests contamination.
5. Membership inference: Statistical tests for whether specific examples were in training
data.
Mitigation:
• Dynamic benchmarks: Regularly generate new test items (LiveCodeBench, Chatbot
Arena)
• Private test sets: Keep test items secret (LMSYS)
• Decontamination during training: Remove detected overlaps from training data
• Report contamination analysis: Disclose overlap metrics alongside benchmark scores
Review: Chapter 14 (LLM Evaluation).
Q: Explain position bias in LLM-as-Judge and how to mitigate it
Answer: Position bias: When using an LLM to judge two responses (A vs B), the model
systematically prefers the response in a particular position (usually first or last), regardless of
quality.
Empirical magnitude: GPT-4 shows 10–15% position bias; Claude shows 5–10%. Smaller
models show larger bias.
Mitigation strategies:
1. Position swapping: Judge each pair twice (A-B and B-A). Final decision = majority. If
disagreement, mark as “tie.” This eliminates systematic position bias but doubles cost.
2. Multi-judge panels: Use 3+ different models as judges. Majority vote reduces individual
model biases.
3. Reference-guided: Provide a rubric or reference answer. Judges score each response
independently against the rubric, then compare scores (eliminates pairwise comparison
entirely).
4. Calibrated prompting: Add explicit instruction: “The order of presentation is random
and should not influence your judgment.”
Additional biases: Verbosity bias (prefers longer responses), self-enhancement bias (models
prefer their own outputs), authority bias (defers to responses that cite sources).
550


<!-- page 551 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Review: Chapter 14 (LLM Evaluation).
27.22
Agentic Memory Questions
Q: Compare the four types of agentic memory and when each is critical
Answer:
Type
What it stores
Access pattern
Critical when
Working
Current
contex-
t/scratchpad
Always in context
Complex multi-step reasoning
Episodic
Past experiences
Retrieved by similar-
ity
Learning from past mistakes
Semantic
Facts and knowledge
Retrieved by concept
Domain-specific tasks
Procedural
Skills and patterns
Triggered
by
task
type
Repeated tool-use
Key insight: These aren’t independent — they interact. Episodic memory feeds semantic memory
(generalizing from episodes to facts). Procedural memory is refined by episodic feedback (learning
which tool sequences work). Working memory orchestrates retrieval from all other types.
MemGPT analogy: Working = hot (in-context), Episodic/Semantic = warm (vector store),
Procedural = cold (archived policies). The agent itself decides when to page information in/out.
Review: Chapter 16 (Agentic Memory Systems).
Q: How does temporal decay work in memory retrieval and why is it important?
Answer: Temporal decay down-weights older memories during retrieval:
score(m) = α · similarity(q, m) + (1 −α) · recency(m)
where recency(m) = e−λ·∆t with ∆t = time since last access.
Why it’s important:
1. Relevance decay: User preferences change. A preference from 6 months ago may be
outdated.
2. Contradiction resolution: When old and new information conflict, recency bias naturally
prefers current truth.
3. Retrieval efficiency: Without decay, memory grows unbounded and retrieval returns
increasingly irrelevant ancient items.
4. Cognitive plausibility: Humans forget too — recent events are more accessible. This
mirrors the spacing effect.
Access-based refresh: When a memory is retrieved and used, its timestamp updates (similar to
LRU caching). Frequently-accessed memories stay “fresh” regardless of creation date.
Decay rate tuning: λ depends on domain. Customer service: high decay (preferences change
fast). Legal/medical: low decay (facts persist). Can be learned via RL.
Review: Chapter 16 (Agentic Memory Systems).
Q: How can RL be used to train memory operations?
Answer: Memory operations (write/read/update/delete) can be actions in the agent’s MDP:
Formulation:
• State: Current context + memory state
• Actions:
Standard
actions
+
memory_write(key/value),
memory_read(query),
memory_delete(key)
551


<!-- page 552 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Reward: Task success (did memory help?) + memory efficiency penalty (fewer reads =
better)
What RL learns:
1. What to store: Important information (API keys/user preferences) vs ephemeral details
2. When to retrieve: Before answering domain questions vs during general chat
3. Compression policy: When to summarize old memories vs keep verbatim
4. Forgetting: When old information is stale and should be removed
Training signal: Counterfactual — “would the agent have succeeded if it hadn’t stored/retrieved
this memory?” Implemented via trajectory comparison: trajectories with good memory use get
higher rewards.
Challenge: Delayed reward — storing information now may only help 100 steps later. Requires
long-horizon credit assignment (GAE with high λ).
Review: Chapters 12 and 16 (LLM Agentic Training; Agentic Memory Systems).
27.23
Agent Orchestration Questions
Q: Explain the context budget problem and how to solve it with dynamic allocation
Answer: The problem: An agent has context window L tokens but needs space for:
C = S + M + T + H + R ≤L
where S = system prompt, M = memory/retrieved context, T = tool descriptions, H = conversation
history, R = reserved for response.
As conversations grow, H increases and pushes out other components.
Dynamic allocation strategy:
1. Fixed minimums: Smin, Rmin are non-negotiable
2. Adaptive history: Summarize old turns when H > Hmax. Keep last k turns verbatim;
summarize the rest.
3. On-demand tools: Only include tool descriptions relevant to current query (not all 50
tools). Use a classifier or embedding similarity to select top-k tools.
4. Lazy memory: Retrieve memory only when needed (after analyzing the query) rather than
pre-loading.
Overflow handling: When total exceeds L even after compression:
• Drop least-important tool descriptions
• Aggressively summarize history to 1-sentence-per-turn
• Reduce memory slots
• If still over: truncate with warning to user
Pre-flight check: Always count tokens BEFORE calling the LLM. Never discover overflow at
inference time.
Review: Chapter 17 (Agent Harness – Context Management and Orchestration).
552


<!-- page 553 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Q: Compare ReAct vs Plan-and-Execute orchestration patterns
Answer:
ReAct (Reason + Act):
• Loop: Thought →Action →Observation →Thought →. . .
• Each step decides the next action based on all previous observations
• Pro: Adaptive — can change direction based on tool outputs
• Con: Myopic — no upfront planning; can get stuck in loops; each LLM call sees entire
history (expensive)
Plan-and-Execute:
• Phase 1: Generate full plan (list of steps)
• Phase 2: Execute steps sequentially (simpler executor; possibly cheaper model)
• Phase 3: Re-plan if execution fails
• Pro: Efficient (planning once is cheaper than reasoning every step); parallelizable independent
steps
• Con: Brittle plans — if early steps fail the plan may be invalid. Re-planning adds latency.
When to use which:
• ReAct: Exploratory tasks; unknown environments; tasks where each step’s result determines
the next
• Plan-and-Execute: Well-defined tasks; known tool set; parallelizable sub-tasks; cost-sensitive
deployments
• Hybrid: Plan at high level then ReAct within each plan step (LangGraph’s recommended
pattern)
Review: Chapters 17 and 18 (Agent Harness; Agent Design Patterns).
Q: How do you detect and prevent infinite loops in agent execution?
Answer: Agents can enter infinite loops when they repeat the same action expecting different
results.
Detection methods:
1. Max iteration guard: Hard limit (e.g., 25 steps). Simple but loses work on genuinely long
tasks.
2. Action hash window: Hash the last k (action/observation) pairs. If current hash matches
a hash from the last w steps then loop detected.
3. Semantic similarity: Embed recent actions. If cosine similarity between consecutive
actions exceeds threshold (>0.95) then likely stuck.
4. Progress monitoring: Define task-specific progress metrics. If no progress in N steps then
intervene.
Recovery strategies:
1. Inject hint: Add system message: “You seem to be repeating actions. Try a different
approach.”
553


<!-- page 554 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
2. Force different action: Mask the repeated action from the action space for the next step.
3. Escalate: Return to user with partial results and ask for guidance.
4. Backtrack: Reset to a checkpoint before the loop began and try alternative path.
Best practice: Combine max iterations (safety net) + hash-based detection (early intervention)
+ graceful escalation (preserve user trust).
Review: Chapters 17 and 18 (Agent Harness; Agent Design Patterns).
554

