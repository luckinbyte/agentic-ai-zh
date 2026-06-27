

<!-- page 565 -->
Chapter 28
Quick Reference
This chapter consolidates key equations, architecture specifications, API references, and failure mode
diagnostics for rapid lookup during development and debugging.
28.1
Core RL & Alignment Equations
PPO Clip:
L = E[min(rt ˆAt, clip(rt, 1±ϵ) ˆAt)],
rt = πθ(at|st)/πold(at|st)
(28.1)
DPO:
L = −E[log σ(β log πθ(yw|x)
πref(yw|x) −β log πθ(yl|x)
πref(yl|x))]
(28.2)
GRPO:
ˆAi = (ri −µG)/σG,
then PPO clip update (no critic)
(28.3)
KTO:
L = λw(1 −v(yw)) + λl · v(yl),
v = σ(β log(πθ/πref) −z)
(28.4)
IPO:
L = E[(log(πθ(yw)/πref(yw)) −log(πθ(yl)/πref(yl)) −1/(2β))2]
(28.5)
ORPO:
L = LSFT(yw) −λ log σ(log(odds(yw)/odds(yl)))
(28.6)
GAE:
ˆAt = PT−t
l=0 (γλ)lδt+l,
δt = rt + γV (st+1) −V (st)
(28.7)
KL Penalty:
Rtotal = rϕ(x, y) −βDKL[πθ(y|x)∥πref(y|x)]
(28.8)
RM (Bradley-Terry):
L = −E[log σ(rϕ(x, yw) −rϕ(x, yl))]
(28.9)
Best-of-N:
y∗= arg
max
yi∼πθ(·|x), i=1..N rϕ(x, yi)
(28.10)
28.2
Transformer & Architecture Formulas
Self-Attention:
Attn(Q, K, V ) = softmax(QK⊤/
p
dk) · V
(28.11)
Multi-Head:
MHA(X) = Concat(head1, . . . , headh)W O,
headi = Attn(XW Q
i , XW K
i , XW V
i )
(28.12)
RoPE:
f(xm, m) = xmeimθj,
θj = 10000−2j/d
(28.13)
LoRA:
W ′ = W0 + (α/r) · BA,
B ∈Rd×r, A ∈Rr×k
(28.14)
KD (soft targets):
LKD = (1−α)LCE(y, ˆy) + α T 2 · KL(pteacher
T
∥pstudent
T
)
(28.15)
FFN (SwiGLU):
FFN(x) = (Swish(xW1) ⊙xW3)W2
(28.16)
565


<!-- page 566 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
28.3
Decoding Methods
Method
Formula / Rule
Key Param
Greedy
yt = arg maxv P(v|y<t)
—
Beam search
Keep top-B partial sequences
by joint probability
B = 4–8
Temperature
P ′(v) = softmax(logitv/T)
T ∈[0.1, 1.5]
Top-k
Zero out all but top-k logits,
renormalize
k = 40–100
Top-p (nucleus)
Keep
smallest
set
V ′
s.t. P
v∈V ′ P(v) ≥p
p = 0.9–0.95
Min-p
Keep tokens with P(v)
≥
pmin · P(vmax)
pmin = 0.05–0.1
Repetition penalty
logitv
←
logitv/θ if v ap-
peared before
θ = 1.1–1.3
28.4
Systems & Parallelism
Formula
Value (70B, BF16)
Description
Model memory
2P bytes
140 GB (weights only)
Adam optimizer
2P × 4 bytes (m + v)
280 GB
Full training footprint
∼8P bytes
560 GB (weights + opt + grad)
FSDP memory/GPU
8P/NGPUs
70 GB with 8 GPUs
Gen arithmetic intensity
2P/2P = 1 FLOP/byte
Heavily memory-bound
Token rate (gen)
HBM_BW /(2P)
∼14 tok/s (A100, batch=1)
TP AllReduce / layer
2 × 2 · T−1
T
· bsd bytes
∼188 MB (70B, TP=8)
PP bubble fraction
(P −1)/(P + M −1)
P=stages, M=micro-batches
MFU
observed_toks
×
6P
/
peak_FLOPS
Target: > 40%
28.5
GPU Hardware Specs
GPU
Memory
BW (HBM)
BF16 TFLOPS
NVLink
Notes
A100-80GB
80 GB HBM2e
2.0 TB/s
312
600 GB/s
Workhorse, widely available
H100-80GB
80 GB HBM3
3.35 TB/s
989
900 GB/s
Current gen, FP8 support
H200-141GB
141 GB HBM3e
4.8 TB/s
989
900 GB/s
Large context / fewer GPUs
B200
192 GB HBM3e
8.0 TB/s
2250
1800 GB/s
Next gen (2025)
566


<!-- page 567 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
28.6
Hyperparameter Ranges
Parameter
Typical Range
Default
Notes
β (DPO/KTO)
0.05–0.5
0.1
Higher = more conservative
ϵ (PPO clip)
0.1–0.3
0.2
Higher = more aggressive updates
γ (GAE discount)
0.99–1.0
1.0
Use 1.0 for episodic tasks
λ (GAE)
0.9–0.99
0.95
Lower = more biased, less variance
KL coeff (βKL)
0.01–0.2
0.05
Auto-adapt to target KL ≈5–8
LR (RLHF)
1e-7 – 5e-6
5e-7
Much lower than pre-training
LR (SFT)
1e-5 – 5e-5
2e-5
Standard fine-tuning range
LoRA rank r
8–128
16–64
Higher r = more capacity, more
memory
LoRA alpha α
r – 2r
2r
Scaling factor; α/r is the effective
scale
Temperature (gen)
0.6–1.0
0.7
Lower = less diverse candidates
Num generations K
4–64
4–16
For GRPO/Online DPO/Best-of-N
Grad clip norm
0.5–2.0
1.0
Prevents gradient explosion
28.7
TRL API Quick Reference
Trainer
Method
Key Config
Data Format
SFTTrainer
Supervised FT
packing,
max_seq_length
prompt + completion
RewardTrainer
Reward model
center_rewards_coefficient
prompt + chosen + rejected
PPOTrainer
PPO
init_kl_coef,
target_kl,
cliprange
prompts (online gen)
DPOTrainer
DPO/IPO
beta,
loss_type="sigmoid"/"ipo"
prompt + chosen + rejected
GRPOTrainer
GRPO
num_generations,
beta, use_vllm
prompts + reward_fn
OnlineDPOTrainer
Online DPO
num_generations,
reward_model_path
prompts (online gen)
KTOTrainer
KTO
desirable_weight,
undesirable_weight
prompt + completion + label
ORPOTrainer
ORPO
beta
prompt + chosen + rejected
Best-of-N (manual)
Best-of-N
n_samples
prompts (inference)
28.8
RAG Pipeline Formulas
Cosine similarity:
sim(q, d) =
q · d
∥q∥· ∥d∥
(28.17)
Retrieval:
Dk = top-kd∈C sim(embed(q), embed(d))
(28.18)
RAG generation:
P(y|q) = PLLM(y | q, Dk)
(28.19)
Chunking overlap:
stride = chunk_size −overlap
(28.20)
Reranker (cross-enc):
score(q, d) = MLP(BERT([q; d]))
(28.21)
567


<!-- page 568 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
28.9
Agentic Design Patterns
Pattern
Structure
Best For
ReAct
Think →Act →Observe →
loop
General tool-use agents
Plan-and-Execute
Plan →Execute steps →Re-
vise
Long-horizon, structured tasks
Supervisor
Router →specialist agents
Multi-domain, clear subtask boundaries
Swarm (handoffs)
Agent transfers control + con-
text
Customer service, escalation flows
Hierarchical
Tree of delegating agents
Complex decomposition
Human-in-the-loop
Agent →Approval gate →
Continue
High-stakes, irreversible actions
28.10
Agent Communication Protocols
Protocol
Scope
Transport
Key Concept
MCP
Tool integration
stdio / HTTP+SSE
Server exposes tools; client discovers
& calls
A2A
Agent-to-agent
HTTP + JSON-RPC
Tasks
with
lifecycle
(submitted→working→done)
OpenAI Function Calling
Tool use
API payload
JSON schema in tools[] array
28.11
Context Window Budget
C ≥
S
|{z}
system
+
M
|{z}
memory/RAG
+
T
|{z}
tool defs
+
H
|{z}
history
+
R
|{z}
reserved output
(28.22)
Rule of thumb for 128K context:
• System prompt: 1–4K tokens (fixed)
• Tool definitions: 2–8K (scales with # tools)
• RAG context: 4–16K (top-k chunks)
• History: grows unbounded →summarize/truncate
• Reserved output: 2–8K
28.12
Common Failure Modes & Fixes
Symptom
Likely Cause
Fix
Reward up, quality down
Reward hacking
RM ensemble, length penalty, increase β
KL exploding (>15)
LR too high or mode collapse
Reduce LR, checkpoint rollback
Entropy collapse
Premature convergence
Increase entropy coeff, raise temperature
Training loss NaN
Gradient explosion
Reduce LR, increase grad clip, check data
No improvement after 5K steps
Bad prompt distribution
Goldilocks filter (20–80% pass rate)
Benchmark regression
Alignment tax
Reduce RL budget, use LoRA, mix SFT data
Length increasing monotonically
Length exploit in RM
Length penalty, retrain RM with length control
OOM during generation
KV cache overflow
Reduce batch, increase TP, PagedAttention
Agent loops forever
No max-iteration guard
Set max_iterations, add loop detection
Tool call parse failures
Inconsistent output format
Few-shot examples, constrained decoding
RAG returns irrelevant docs
Poor embedding / chunking
Reranker, hybrid search, smaller chunks
Multi-agent deadlock
Circular dependencies
DAG enforcement, timeout per agent
568


<!-- page 569 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
28.13
Method Selection Decision Tree
1. Have paired preferences (chosen + rejected)?
• Noisy labels →IPO
• Memory-constrained, no SFT done yet →ORPO
• Clean data, limited compute →DPO
• DPO plateaus, want exploration →Online DPO
2. Have only binary feedback (thumbs up/down)? →KTO
3. Have verifiable rewards (math/code)? →GRPO
4. Need maximum quality, any cost? →PPO
5. Want training-free improvement? →Best-of-N
28.14
Evaluation Metrics
Metric
Range
What It Measures
Perplexity
[1, ∞)
Model’s surprise; lower = better language mod-
eling
Win Rate (vs. baseline)
[0, 1]
Fraction of outputs preferred by judge/human
BLEU
[0, 1]
n-gram overlap with reference (precision-focused)
ROUGE-L
[0, 1]
Longest common subsequence with reference
Pass@k
[0, 1]
Probability ≥1 of k code samples passes tests
MMLU / GPQA
[0, 1]
Multi-choice accuracy on knowledge/reasoning
benchmarks
HumanEval
[0, 1]
Functional correctness of generated code
Faithfulness (RAG)
[0, 1]
Fraction of claims supported by retrieved context
Context Relevancy
[0, 1]
Fraction of retrieved content relevant to query
Answer Relevancy
[0, 1]
Degree to which answer addresses the question
28.15
Reasoning & Test-Time Scaling
Method
Compute Cost
Mechanism
Chain-of-Thought (CoT)
1.5–3× tokens
“Think step by step” in prompt
Self-Consistency
N× generation
Sample N CoT paths, majority vote on final
answer
Tree-of-Thought (ToT)
B × D× generation
BFS/DFS over reasoning tree; evaluate branches
Best-of-N
N× generation
Sample N, score with RM, pick highest
Beam search (on reasoning)
B× generation
Maintain top-B partial reasoning chains
Budget forcing
Variable
Allocate more tokens to harder problems dynam-
ically
Verification (ORM/PRM)
N× gen + scoring
Generate N solutions, rank by outcome/process
RM
569


<!-- page 570 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
28.16
Memory System Types
Type
Storage
Use Case
Working memory
Context window
Current conversation, immediate tool results
Episodic memory
Vector store
Past interactions, user preferences, session his-
tory
Semantic memory
Knowledge graph / embed-
dings
Facts, concepts, domain knowledge
Procedural memory
Skill library / code
How-to procedures, learned workflows
28.17
MCP Quick Reference
Primitive
Direction
Side Effects?
Purpose
Tools
Client →Server
Yes
Execute actions (create,
modify,
delete)
Resources
Client →Server
No (read-only)
Read data (files, DB records, con-
figs)
Prompts
Client →Server
No
Reusable templates for common
tasks
Sampling
Server →Client
No
Server requests LLM generation
from client
Transport: stdio (local subprocess) or HTTP+SSE (remote, streamable).
Discovery: Client calls tools/list, resources/list, prompts/list at connection init.
Tool annotations: readOnlyHint, destructiveHint, idempotentHint, openWorldHint.
28.18
A2A Protocol Quick Reference
Concept
Description
Agent Card
JSON at /.well-known/agent.json — name, skills, supported
content types
Task
Unit of work: id, status, artifacts. Lifecycle: submitted →
working →completed/failed
Message
Communication unit within a task (role: user/agent, parts: text/-
file/data)
Artifact
Output produced by the agent (structured data, files, generated
content)
Push Notifications
Webhook-based
updates
for
long-running
tasks
(via
tasks/pushNotification/set)
Key endpoints: tasks/send (create/update), tasks/get (poll status), tasks/sendSubscribe
(SSE stream).
570


<!-- page 571 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
28.19
Agent Framework Comparison
Framework
Orchestration
Multi-Agent
Best For
LangGraph
Explicit state graph
Conditional routing
Production: persistence, HITL, fine
control
OpenAI Agents SDK
Declarative handoffs
Handoff-based
Simplicity: guardrails, tracing, fast
start
AutoGen (AG2)
Conversation-driven
GroupChat
Prototyping:
code execution, re-
search
CrewAI
Role-based teams
Sequential/parallel
Low-code:
quick demos,
simple
pipelines
Google ADK
Session + events
A2A native
Enterprise:
artifact mgmt, multi-
modal
28.20
Agentic RL Formulas
Trajectory GRPO:
ˆAi = (R(τi) −µG)/σG,
R(τi) =
X
t
r(τi)
t
(28.23)
Agent reward:
R = w1Rtask + w2Refficiency + w3Rsafety,
Reff = max(0, 1 −steps/Nmax)
(28.24)
Masking:
L =
X
t∈agent tokens
min(rt ˆAt, clip(rt) ˆAt)
(mask env outputs)
(28.25)
Pass@k :
1 −
 n−c
k

 n
k
 ,
n = total samples, c = correct
(28.26)
28.21
Agent Security Checklist
Threat
Layer
Mitigation
Prompt injection (direct)
Input
Input validation, instruction hierarchy, delimiters
Prompt injection (indirect)
Tool output
Treat tool output as untrusted; don’t follow in-
structions in retrieved docs
Tool misuse
Execution
Least-privilege permissions; destructiveHint
gates; sandboxing
Data exfiltration
Output
Output filtering; restrict tool access to allowed
domains
Excessive autonomy
Architecture
Max iterations; cost budgets; human approval
gates
Confused deputy
Multi-agent
Verify task origin; capability-based access control
571


<!-- page 572 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
28.22
Agent Evaluation Metrics
Metric
Formula / Definition
Target
Task Success Rate (TSR)
Correct completions / total
tasks
> 85% (production)
Steps to completion
Avg agent actions per success-
ful task
Lower = more efficient
Cost per task
Total tokens × price/token
Budget-dependent
Latency (TTFC)
Time from request to first use-
ful output
< 5s for interactive
Tool call accuracy
Correct tool selections / total
calls
> 90%
Recovery rate
Successful retries / initial fail-
ures
> 60%
Human escalation rate
Tasks requiring human / total
tasks
< 15%
28.23
Key Agentic Benchmarks
Benchmark
Domain
Metric
SOTA (2025)
SWE-bench Verified
Software engineering
% resolved issues
∼70%
WebArena
Web browsing
Task success rate
∼40%
OSWorld
Desktop
computer
use
Task success rate
∼25%
GAIA
General AI assistant
Exact match accu-
racy
∼75% (L1)
Tau-bench
Tool-use reliability
Pass rate (5 trials)
∼65%
HumanEval / MBPP
Code generation
Pass@1
> 95%
572
