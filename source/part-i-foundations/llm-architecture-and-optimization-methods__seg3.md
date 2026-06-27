<!-- page 54 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
How NLAEs work.
1. Encoder: A language model reads the hidden activations (or the input text) and produces a
natural language description of the active concepts: e.g., “The text discusses French cuisine
and uses formal academic tone.”
2. Decoder: A second language model reads the natural language description and reconstructs
the original activations (or predicts the next token).
3. Training: Both encoder and decoder are trained end-to-end to minimize reconstruction loss,
with the bottleneck being a variable-length natural language string rather than a sparse vector.
Advantages over SAEs.
• Self-interpreting: Features are literally natural language—no manual labelling needed.
• Compositional: Can express complex, relational concepts (“a sarcastic response to a factual
claim”) that SAE features cannot represent as single directions.
• Hierarchical: Descriptions can capture both fine-grained (word-level) and coarse (document-
level) properties in the same representation.
• Auditable: The bottleneck description is human-readable, enabling direct inspection of what
information the model “thinks” is present.
Limitations.
NLAEs introduce a language-model-in-the-loop, making them computationally ex-
pensive and potentially subject to the same faithfulness concerns as any model-generated explanation.
They also cannot easily represent sub-symbolic features (geometric patterns, exact numerical values)
that SAEs handle naturally as activation magnitudes.
The Interpretability Stack
Think of interpretability tools as a hierarchy:
1. Attention maps: “What is the model looking at?” (cheapest, least faithful)
2. Probing classifiers: “What information is encoded at this layer?”
3. Sparse Autoencoders: “What monosemantic features are active?” (scalable, requires
human labelling)
4. Natural Language Autoencoders: “What does the model think is happening?” (self-
interpreting, expensive)
5. Causal tracing / patching: “Which components actually cause this output?” (most
faithful, most expensive)
Each level trades off between cost, scalability, and faithfulness of explanation.
1.4
Prediction Heads: What Transformers Output
The transformer body produces contextual hidden states ht ∈Rd for each position. What we do
with these hidden states—the prediction head—defines the task. The same transformer backbone
can serve radically different purposes simply by swapping the head.
1.4.1
Language Modeling Head (Pretraining)
The standard LM head projects the final hidden state to vocabulary logits and trains with cross-
entropy loss over the next token:
54


<!-- page 55 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Figure 1.7: The same transformer backbone supports different tasks by swapping the prediction head. All
three heads used in this paper share identical architecture below the final projection layer.
P(xt+1|x≤t) = softmax(Whead · ht + b)
(1.7)
where Whead ∈R|V|×d (often tied with the embedding matrix: Whead = ET ).
LM Head Properties
• Training objective: Causal language modeling (predict next token for every position)
• Loss: LLM = −1
T
PT
t=1 log P(xt|x<t)
• Label: Every token is both input (shifted right) and target (shifted left)
• Used during: Pretraining on large corpora (trillions of tokens)
• Key insight: The model learns general language understanding as a byproduct of next-token
prediction
1.4.2
Conditional Generation Head (SFT / Instruction Following)
For supervised fine-tuning (SFT), the architecture is identical to the LM head—the same linear
projection to vocabulary logits. The difference is purely in what we compute loss on:
LSFT = −1
|y|
|y|
X
t=1
log P(yt|xprompt, y<t)
(1.8)
Conditional Head – Key Differences from LM Head
• Loss masking: Only compute loss on the response tokens, not the prompt/instruction. The
prompt provides context but no gradient signal.
• Conditioning: The model learns to generate responses conditioned on specific instruction
formats (system prompts, user queries, tool calls).
• Format tokens: Special tokens (<|user|>, <|assistant|>) guide the model to produce
structured outputs.
• Used during: SFT on curated instruction-response pairs; also during RL policy generation
(the policy head that produces actions/responses).
Same Head – Different Training Signal
The LM head and SFT head are architecturally identical (same Whead). The only difference is
that during SFT, we mask the loss on prompt tokens. This subtle change transforms a general text
predictor into a instruction-following assistant. The head learns to “activate” different generation
modes based on the conditioning context.
55


<!-- page 56 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.4.3
Value Head (Regression for RL)
In reinforcement learning (PPO, GRPO), we need to estimate how good a state is—this requires a
scalar output, not vocabulary logits. The value head replaces the LM projection with a simple
regression layer:
V (st) = wT
value · ht + b ∈R
(1.9)
where wvalue ∈Rd and b ∈R.
Value Head Properties
• Output: Single scalar (expected cumulative reward from this state)
• Loss: MSE between predicted and actual returns: LV = 1
T
P
t(V (st) −Rt)2
• Architecture: Linear layer Rd →R1 (sometimes with a small MLP: d →256 →1)
• Backbone sharing: Often shares the transformer body with the policy (with a separate
value head), or uses a completely separate critic network
• Used during: PPO advantage estimation (GAE), reward model scoring
1.4.4
Head Selection Summary
Table 1.9: Prediction heads used throughout this paper and their training contexts.
Head
Output
Loss
Stage
Purpose
LM Head
R|V|
Cross-entropy
(all tokens)
Pretraining
Learn language
from raw text
Conditional Head
R|V|
Cross-entropy
(response only)
SFT
Learn to follow
instructions
Value Head
R1
MSE
RL (PPO)
Estimate
state
value for advan-
tage
Reward Head
R1
Pairwise ranking
RM training
Score
response
quality
Head Initialization Matters
When adding a value head to a pretrained LM, initialize it near zero (small random weights).
If initialized with large values, the initial value estimates will be wildly wrong, causing huge
advantages and unstable PPO updates. Common practice: initialize the final linear layer with
N(0, 1/
√
d) or simply zeros.
1.4.5
HuggingFace Implementation
from
transformers
import (
AutoModelForCausalLM ,
# LM head (pretraining + SFT)
AutoModelForSequenceClassification ,
# Reward
head
AutoTokenizer ,
)
from trl import
AutoModelForCausalLMWithValueHead
# Value
head (PPO)
import
torch
model_name = "meta -llama/Llama -3.1 -8B-Instruct"
tokenizer = AutoTokenizer. from_pretrained (model_name)
# === 1. LM Head (Pretraining / SFT) ===
# The default
CausalLM
model
-- projects
hidden
states to vocab
logits
lm_model = AutoModelForCausalLM . from_pretrained (
56


<!-- page 57 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
model_name ,
torch_dtype=torch.bfloat16 ,
device_map="auto",
)
# lm_model.lm_head: Linear(hidden_size
-> vocab_size)
# Output: logits of shape (batch , seq_len , vocab_size)
inputs = tokenizer("The
capital of France is", return_tensors ="pt")
outputs = lm_model (** inputs)
next_token_logits = outputs.logits [:, -1, :]
# (batch , vocab_size)
probs = torch.softmax(next_token_logits , dim=-1)
# === 2. Conditional
Head (SFT) ===
# Architecturally
identical to LM head -- difference is in loss
masking
# During SFT , we only
compute
loss on response
tokens:
messages = [
{"role": "user", "content": "What is 2+2?"},
{"role": "assistant", "content": "4"},
]
formatted = tokenizer. apply_chat_template (messages , return_tensors ="pt")
labels = formatted.clone ()
# Mask
prompt
tokens (set to
-100 so cross -entropy
ignores
them)
prompt_len = len(tokenizer. apply_chat_template (messages [:1]))
labels [:, :prompt_len] =
-100
loss = lm_model(input_ids=formatted , labels=labels).loss
# === 3. Value
Head (PPO Critic) ===
# Adds a Linear(hidden_size
-> 1) on top of the LM backbone
value_model = AutoModelForCausalLMWithValueHead . from_pretrained (
model_name ,
torch_dtype=torch.bfloat16 ,
device_map="auto",
)
# value_model.v_head: Linear(hidden_size
-> 1)
# Returns
both LM logits AND per -token
value
estimates
inputs = tokenizer("Explain
quantum
computing", return_tensors ="pt")
lm_logits , loss , values = value_model(
**inputs , return_dict=False
)
# values
shape: (batch , seq_len , 1) -- scalar
estimate
per token
# === 4. Reward
Head (Reward
Model) ===
# Classification
head: Linear(hidden_size
-> 1) on last
token
reward_model = AutoModelForSequenceClassification . from_pretrained (
model_name ,
num_labels =1,
# single
scalar
output
torch_dtype=torch.bfloat16 ,
device_map="auto",
)
# Scores
entire
sequence by pooling
the last
token ’s hidden
state
inputs = tokenizer("Good
response
here", return_tensors ="pt")
reward_score = reward_model (** inputs).logits
# shape: (batch , 1)
Listing 1.3: Loading and using different prediction heads with HuggingFace.
Weight Tying: LM Head = Embedding Matrix Transposed
Most modern LLMs tie the LM head weights with the input embedding matrix: lm_head.weight =
model.embed_tokens.weight. This means the LM head is not a separately learned layer—it reuses
the embedding table. Benefits: fewer parameters (|V| × d saved), better generalization, and the
geometric structure of the embedding space directly determines token probabilities. You can verify
this in HuggingFace: model.lm_head.weight is model.model.embed_tokens.weight returns
True for most models.
57


<!-- page 58 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.5
Optimization Theory for LLM Training
Training a large language model means finding the set of parameters θ (billions of weights) that
minimizes the loss function L(θ) — typically the negative log-likelihood of the next token. This is an
optimization problem in extraordinarily high-dimensional space, and the algorithm used to navigate
this space determines whether training succeeds, diverges, or stalls.
1.5.1
Gradient Descent: The Foundation
What is a Gradient?
The gradient ∇θL is a vector that points in the direction of steepest increase
of the loss. Each component ∂L
∂θi tells us how much the loss would change if we slightly increased
parameter θi. To decrease the loss, we move in the opposite direction:
θt+1 = θt −η∇θL(θt)
(1.10)
where η > 0 is the learning rate — the step size. This is gradient descent [77].
Figure 1.8: Gradient descent: starting from a random initialization θ0, each step moves the parameters in
the direction that reduces the loss, with step size controlled by the learning rate η. The process converges
toward a (local) minimum.
Why Full Gradient Descent is Impractical.
Computing the exact gradient requires evaluating
the loss over the entire training dataset (trillions of tokens for LLMs). This is computationally
prohibitive — a single gradient step would require a full pass over all data.
Stochastic Gradient Descent (SGD).
The solution: estimate the gradient from a small random
subset (mini-batch) of the data [78]:
∇θL(θ) ≈1
B
B
X
i=1
∇θℓ(θ; xi)
where B is the batch size (typically 1K–4M tokens for LLMs). The mini-batch gradient is a noisy
but unbiased estimate of the true gradient.
Why Mini-Batch SGD Works
• Computational efficiency: Each step costs O(B) instead of O(Ntotal). With B = 4096
tokens and 15T total tokens, each step is ∼4 billion× cheaper.
• Noise as regularization: The stochastic noise helps escape sharp local minima, finding
flatter regions that generalize better.
• GPU utilization: Mini-batches are large enough to saturate GPU parallelism (matrix
58


<!-- page 59 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
multiplications become compute-bound rather than memory-bound).
• Convergence: Theoretically converges to a local minimum at rate O(1/
√
T) (slower than
exact GD’s O(1/T), but each step is millions of times cheaper).
From SGD to Adaptive Methods.
While SGD with momentum works well for vision models
(CNNs), LLM training requires adaptive optimizers — algorithms that maintain a per-parameter
learning rate.
1.5.2
Why Vanilla SGD Fails for LLMs
Stochastic Gradient Descent updates weights as:
θt+1 = θt −η∇θL(θt)
SGD Problems for LLMs
• Different gradient scales per layer: Early layers in a transformer have much smaller
gradients than later layers (vanishing gradients). A single learning rate η is too large for
some parameters and too small for others.
• Sparse gradients: Embedding layers receive gradients only for tokens in the current batch.
Most embedding rows have zero gradient. SGD with momentum wastes momentum on
zero-gradient rows.
• Saddle points: High-dimensional loss landscapes have many saddle points. SGD can stall;
adaptive methods escape faster.
• Sensitivity to learning rate: SGD requires careful tuning; a 2× change in η can cause
divergence.
1.5.3
Adam – Adaptive Moment Estimation
Adam [79] maintains per-parameter estimates of the first moment (mean of gradients) and second
moment (uncentered variance of gradients).
Adam Update Equations
Given gradient gt = ∇θL(θt), hyperparameters β1, β2, ϵ, η:
Step 1 – Update biased first moment estimate:
mt = β1mt−1 + (1 −β1)gt
Step 2 – Update biased second moment estimate:
vt = β2vt−1 + (1 −β2)g2
t
Step 3 – Bias correction:
ˆmt =
mt
1 −βt
1
,
ˆvt =
vt
1 −βt
2
Step 4 – Parameter update:
θt+1 = θt −η ·
ˆmt
√ˆvt + ϵ
Typical values: β1 = 0.9, β2 = 0.95 or 0.999, ϵ = 10−8, η = 10−4 to 10−5.
59


<!-- page 60 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
What Each Term Does
• mt (momentum): Exponential moving average of gradients. Smooths out noisy gradient
estimates. β1 = 0.9 means the current gradient contributes 10% and the history contributes
90%.
• vt (adaptive LR): EMA of squared gradients. Parameters with consistently large gradients
get a smaller effective learning rate (η/√vt). Parameters with small gradients get a larger
effective LR. This is the key to handling different gradient scales per layer.
• ˆmt, ˆvt (bias correction): At t = 1, m1 = (1 −β1)g1 is much smaller than the true mean.
Dividing by (1 −βt
1) corrects this initialization bias. Without it, early steps are too small.
• ϵ (numerical stability): Prevents division by zero. Also acts as a floor on the effective
learning rate.
1.5.4
AdamW – Decoupled Weight Decay
AdamW [80] fixes a subtle but important issue with how weight decay interacts with adaptive
optimizers.
Why L2 Regularization ̸= Weight Decay in Adam
With L2 regularization, the loss becomes L + λ
2∥θ∥2, so the gradient is gt + λθt. In Adam, this
regularization gradient is scaled by the adaptive factor 1/√ˆvt:
θt+1 = θt −η · ˆmt + λθt
√ˆvt + ϵ
Parameters with large vt (large gradient variance) get less regularization. This is not what we
want – weight decay should be uniform.
AdamW – Decoupled Weight Decay
AdamW (Loshchilov & Hutter, 2017) applies weight decay directly to the parameters, outside the
adaptive scaling:
θt+1 = θt −η ·
ˆmt
√ˆvt + ϵ −ηλθt
The weight decay term ηλθt is not divided by √ˆvt. This gives uniform regularization across all
parameters regardless of their gradient history.
Typical value: λ = 0.1 for LLM training.
Always Use AdamW – Never Plain Adam – for LLMs
The difference between Adam and AdamW is subtle but matters. With Adam + L2, the effective
weight decay is stronger for parameters with small gradient variance (e.g., biases, LayerNorm
parameters) and weaker for parameters with large gradient variance (e.g., attention weights).
AdamW gives the intended uniform regularization. Most frameworks default to AdamW; double-
check your optimizer class.
60


<!-- page 61 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.5.5
Learning Rate – The Most Important Hyperparameter
Typical Learning Rates by Training Phase
Phase
Typical LR
Notes
Pretraining (from scratch)
1e-4 to 3e-4
Large model, large batch
Continued pretraining
1e-5 to 1e-4
Smaller LR to preserve knowledge
SFT (supervised fine-tune)
1e-5 to 2e-5
Standard range
LoRA fine-tuning
1e-4 to 3e-4
Higher LR for adapter weights
For RL learning rates (PPO, DPO, GRPO) see §11.15.
1.5.6
Learning Rate Warmup
Why Warmup is Necessary
At the start of training, vt (the second moment estimate) is initialized to zero. After bias correction:
ˆvt = vt/(1−βt
2). At t = 1 with β2 = 0.999: ˆv1 = v1/(1−0.999) = 1000v1. This means the effective
learning rate is η/√1000v1 – much smaller than intended.
However, if the first gradient is unusually large (common at initialization), the second moment
estimate can be dominated by this outlier, causing erratic early steps. Warmup mitigates this by
starting with a very small LR and gradually increasing it, giving vt time to accumulate a reliable
estimate.
• Linear warmup: ηt = ηmax × t/Twarmup
• Typical warmup duration: 1–5% of total steps for pretraining; 3–10% for fine-tuning (shorter
runs need proportionally more warmup)
• For SFT: 50–200 warmup steps is typical
1.5.7
Learning Rate Schedules
Figure 1.9: Common learning rate schedules. All include a linear warmup phase. WSD (Warmup-Stable-
Decay) is the emerging standard for pretraining.
(a) Constant.
Simplest schedule. Good for short fine-tuning runs where you want to avoid
over-decaying the LR. Risk: no annealing means the model may not converge to the sharpest
minimum.
61


<!-- page 62 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
(b) Cosine Decay.
ηt = ηmin + 1
2(ηmax −ηmin)
 
1 + cos
 
t −Twarmup
T −Twarmup
π
!!
Standard for pretraining and SFT. Smooth decay avoids abrupt LR changes. ηmin is typically
ηmax/10.
(c) Linear Decay.
Simpler than cosine, similar empirical results. Preferred when you want
predictable LR at any step.
(d) WSD – Warmup-Stable-Decay.
The new standard for large-scale pretraining [25, 81]. Three
phases:
1. Warmup: Linear ramp to ηmax (1–5% of steps)
2. Stable: Constant ηmax for the majority of training
3. Decay: Fast cosine or linear decay to ηmin (last 10–20% of steps)
Key advantage: the stable phase allows checkpointing at any point and continuing training. The
decay phase can be applied at the end of any run.
(e) Cosine with Restarts (SGDR).
Periodic restarts reset the LR to ηmax. Can help escape
local minima. Less common for LLMs; more useful for smaller models.
1.5.8
Gradient Clipping
Gradient Clipping
Gradient clipping rescales the gradient if its global norm exceeds a threshold:
gt ←gt · min

1,
τ
∥gt∥2

where τ is max_grad_norm (typically 1.0).
Gradient Clipping vs. LR Reduction
Gradient clipping and reducing the learning rate both limit the size of parameter updates. The
difference: clipping preserves the direction of the gradient (just scales the magnitude), while a
smaller LR scales all updates uniformly. Clipping is better for handling occasional large gradients
without slowing down normal training steps.
Putting It Together: HuggingFace Optimizer Configuration
The following snippet shows how the concepts from this section—AdamW with decoupled weight
decay (§1.6.6), cosine learning-rate scheduling with linear warmup (§1.6.7), and gradient clipping
(§1.6.8)—come together in practice using the HuggingFace transformers library.
from
transformers
import
TrainingArguments , Trainer
from
transformers
import
get_cosine_schedule_with_warmup
import
torch
# --- Option 1: Using
TrainingArguments (recommended) ---
training_args = TrainingArguments (
output_dir="./ checkpoints",
# AdamW
optimizer (decoupled
weight decay , S1 .6.6)
optim="adamw_torch",
62


<!-- page 63 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
learning_rate =2e-5,
# peak LR after
warmup
adam_beta1 =0.9,
# first
moment
decay
adam_beta2 =0.999 ,
# second
moment
decay
adam_epsilon =1e-8,
# numerical
stability
weight_decay =0.01 ,
# decoupled L2 penalty
# Learning
rate
schedule (S1 .6.7)
lr_scheduler_type ="cosine",
# cosine
decay to 0
warmup_ratio =0.1,
# 10% of steps = linear
warmup
# Gradient
clipping (S1 .6.8)
max_grad_norm =1.0,
# clip by global L2 norm
# Mixed
precision (S1 .6.9)
bf16=True ,
# use
BFloat16 on Ampere+ GPUs
# Training
duration
num_train_epochs =3,
per_device_train_batch_size =8,
gradient_accumulation_steps =4,
# effective
batch = 8*4 = 32
)
trainer = Trainer(
model=model ,
args=training_args ,
train_dataset=dataset ,
)
trainer.train ()
# --- Option 2: Manual
control (for custom
training
loops) ---
from
torch.optim
import
AdamW
# Separate
weight -decay
groups (don’t regularize
biases/norms)
no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
param_groups = [
{
"params": [p for n, p in model. named_parameters ()
if not any(nd in n for nd in no_decay)],
"weight_decay": 0.01,
},
{
"params": [p for n, p in model. named_parameters ()
if any(nd in n for nd in no_decay)],
"weight_decay": 0.0,
},
]
optimizer = AdamW(param_groups , lr=2e-5, betas =(0.9 , 0.999))
# Cosine
schedule
with
linear
warmup
total_steps = len( train_dataloader ) * num_epochs
warmup_steps = int (0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup (
optimizer ,
num_warmup_steps =warmup_steps ,
num_training_steps =total_steps ,
)
# Training
loop with
gradient
clipping
for batch in train_dataloader :
outputs = model (** batch)
loss = outputs.loss
loss.backward ()
# Clip
gradients
before
optimizer
step
torch.nn.utils. clip_grad_norm_ (model.parameters (), max_norm =1.0)
63


<!-- page 64 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
optimizer.step ()
scheduler.step ()
optimizer.zero_grad ()
Listing 1.4: Complete optimizer configuration combining AdamW – cosine schedule – and gradient clipping.
Practical Tips
• Weight decay exclusion: bias terms and layer-norm weights should not be regularized—
they have few parameters and regularizing them hurts performance [80].
• Warmup ratio: 5–10% of total steps is standard; too little warmup with a high LR can
destabilize early training.
• Gradient accumulation: simulates larger batches on limited GPU memory; clipping
applies to the accumulated gradient.
• BF16 vs. FP16: prefer bf16=True on Ampere+ GPUs (wider dynamic range avoids loss
scaling); fall back to fp16=True on older hardware.
1.5.9
Mixed Precision Training
BF16 vs. FP16
Format
Exponent bits
Mantissa bits
Dynamic range
FP32
8
23
∼10−38 to 1038
BF16
8
7
Same as FP32 (same ex-
ponent)
FP16
5
10
∼6 × 10−5 to 65504
BF16 Over FP16: Why Range Beats Precision in LLM Training
BF16 has the same exponent range as FP32, so it can represent the same range of values (just
with less precision in the mantissa). FP16 has a much smaller dynamic range – gradients or
activations that exceed 65504 cause overflow (NaN/Inf). This is why FP16 training requires loss
scaling (multiplying the loss by a large constant to keep gradients in FP16 range), while BF16
training typically does not. A100 and H100 support BF16 natively; use BF16 unless you have a
specific reason for FP16.
Loss Scaling (FP16 only).
1. Multiply loss by scale factor S (e.g., S = 215)
2. Compute gradients in FP16 (scaled by S)
3. Before optimizer step, divide gradients by S
4. Check for overflow (NaN/Inf); if found, skip step and reduce S
5. If no overflow for N consecutive steps, increase S
FP32 Master Weights.
In mixed precision training, weights are stored in FP32 (master copy)
and cast to BF16/FP16 for the forward/backward pass. The optimizer step is done in FP32. This is
important because:
• Small gradient updates (∆θ ≪θ) would be lost in BF16 precision (7 mantissa bits ≈0.8%
relative precision)
64


<!-- page 65 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• FP32 master weights ensure accurate accumulation of small updates over many steps
• Memory cost: 2× weight storage (FP32 + BF16 copy)
When FP32 Master Weights Are Critical
FP32 master weights are most important for:
• Long training runs (many small gradient steps accumulate)
• Small learning rates (updates are tiny relative to weight magnitude)
For short SFT runs with large LR, BF16-only training (no FP32 master weights) often works fine
and saves memory. For RL training, FP32 master weights are essential—see §11.15.
Mixed Precision in Practice: HuggingFace
# ===
HuggingFace
TrainingArguments (simplest
approach) ===
from
transformers
import
TrainingArguments
# BF16 on Ampere+ GPUs (A100 , H100 , RTX 30xx/40xx)
args_bf16 = TrainingArguments (
output_dir="./out",
bf16=True ,
# BF16
forward/backward; FP32
master
weights
bf16_full_eval =True ,
# also use BF16
during
evaluation
# No loss
scaling
needed
-- BF16 has FP32 -equivalent
range
)
# FP16 on older
GPUs (V100 , T4 , RTX 20xx)
args_fp16 = TrainingArguments (
output_dir="./out",
fp16=True ,
# FP16
forward/backward
fp16_full_eval =False ,
# keep eval in FP32 for
accuracy
# Loss
scaling is automatic
via
PyTorch
GradScaler
)
# === Manual
PyTorch
AMP (for custom
training
loops) ===
import
torch
# Setup (PyTorch 2.x API)
use_fp16 = not torch.cuda. is_bf16_supported ()
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
# only
needed for FP16
optimizer = torch.optim.AdamW(model.parameters (), lr=2e-5)
dtype = torch.float16 if use_fp16
else
torch.bfloat16
for batch in train_dataloader :
optimizer.zero_grad ()
# Autocast: run
forward
pass in reduced
precision
with
torch.autocast("cuda", dtype=dtype):
outputs = model (** batch)
loss = outputs.loss
if use_fp16:
# FP16 path: scale
loss to prevent
gradient
underflow
scaler.scale(loss).backward ()
scaler.unscale_(optimizer)
# unscale
before
clipping
torch.nn.utils. clip_grad_norm_ (model.parameters (), 1.0)
scaler.step(optimizer)
# skips
step on overflow
scaler.update ()
# adjust
scale
factor
else:
# BF16 path: no scaling
needed
loss.backward ()
torch.nn.utils. clip_grad_norm_ (model.parameters (), 1.0)
65


<!-- page 66 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
optimizer.step ()
scheduler.step ()
Listing 1.5: Mixed precision training with HuggingFace and manual PyTorch AMP.
Key Differences: BF16 vs. FP16 in Code
• BF16: just wrap with autocast(dtype=torch.bfloat16)—no scaler needed. Simpler code
and more numerically stable.
• FP16: requires GradScaler to prevent gradient underflow. The scaler dynamically adjusts
a multiplier; if overflow is detected (NaN), the optimizer step is skipped and the scale is
reduced.
• Gradient clipping + FP16:
you must call scaler.unscale_(optimizer) before
clip_grad_norm_, otherwise you’re clipping scaled gradients (wrong threshold).
• Memory savings: % reduction in activation memory (activations stored in 16-bit); weight
memory depends on whether you keep FP32 master copies.
1.5.10
Practical Optimizer Settings by Training Phase
Optimizer Hyperparameter Reference Table
Phase
Optimizer
LR
WD
Warmup
Schedule
Pretraining
AdamW
3e-4
0.1
2000 steps
WSD or Co-
sine
SFT
AdamW
2e-5
0.01
100 steps
Cosine
LoRA SFT
AdamW
2e-4
0.01
100 steps
Cosine
All use: β1=0.9, β2=0.95, ϵ=10−8, max_grad_norm=1.0, BF16. For RL settings see §11.15.
Diagnosing Training Instability
# Monitor
these
metrics to diagnose
optimizer
issues:
# 1. Gradient
norm -- should be < max_grad_norm
most of the time
# 2. Loss
scale (FP16) -- should be stable , not
constantly
decreasing
# 3. Parameter
update
norm -- should be << parameter
norm
import
torch
def
log_optimizer_stats (model , optimizer , step):
# Gradient
norm (before
clipping)
total_norm = 0.0
for p in model.parameters ():
if p.grad is not None:
total_norm += p.grad.data.norm (2).item () ** 2
total_norm = total_norm ** 0.5
# Adam
second
moment
stats (proxy for
adaptive LR)
v_norms = []
for group in optimizer.param_groups :
for p in group[’params ’]:
state = optimizer.state[p]
if ’exp_avg_sq ’ in state:
v_norms.append(state[’exp_avg_sq ’]. mean ().item ())
print(f"Step {step }: grad_norm ={ total_norm :.3f}, "
f"mean_v ={sum(v_norms)/len(v_norms):.6f}")
# Red flags:
66


<!-- page 67 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
# grad_norm
>> 1.0
repeatedly
-> reduce LR or increase
warmup
# grad_norm == 0.0 -> gradient
vanishing or wrong
loss
# loss_scale
decreasing
-> FP16 overflow , switch to BF16
# v very
small
-> Adam not warmed up yet , extend
warmup
The Learning Rate is the Most Important Hyperparameter
In practice, getting the learning rate right matters more than any other hyperparameter. A rule
of thumb for LLM fine-tuning:
• Start with the values in the table above
• If loss diverges (increases after initial decrease): LR is too high
• If loss decreases very slowly and plateaus early: LR is too low
• If loss is unstable (oscillates): LR is too high or warmup is too short
The second most important hyperparameter is batch size (affects gradient noise and effective LR
via the linear scaling rule). Everything else is secondary.
1.6
Flash Attention – Algorithm and Hardware Awareness
Flash Attention [7, 82] is one of the most impactful algorithmic innovations in deep learning since
the transformer itself. It does not change the mathematical result of attention – it computes exactly
the same output – but it restructures the memory access pattern so that the GPU’s limited fast
SRAM does all the heavy lifting, cutting HBM footprint from O(n2) to O(n) and delivering 2–4×
end-to-end wall-clock gains on typical workloads.
1.6.1
The Standard Attention Memory Problem
Standard scaled dot-product attention is:
Attention(Q, K, V ) = softmax
 
QKT
√dk
!
V
Standard Attention Memory Complexity
For sequence length n and head dimension d:
• Q, K, V ∈Rn×d: O(nd) memory
• S = QKT ∈Rn×n: O(n2) memory – the bottleneck
• P = softmax(S) ∈Rn×n: another O(n2)
• O = PV ∈Rn×d: O(nd)
At n = 8192, d = 128, BF16: the attention matrix alone is 81922 × 2 ≈134 MB per head. With 32
heads, that is 4.3 GB just for one layer’s attention scores.
Why O(n2) is Catastrophic
The attention matrix must be written to HBM (it does not fit in SRAM for long sequences), then
read back for the softmax, then read again for the PV product. Each of these HBM round-trips
is slow. For n = 32768 (32K context), the attention matrix is 327682 × 2 ≈2 GB per head –
completely infeasible to store.
67


<!-- page 68 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.6.2
The Flash Attention Key Insight – Tiling and Online Softmax
The core insight is: we never need the full n × n matrix in memory at once. We can compute
the output O block-by-block if we use the online softmax trick.
Online Softmax.
Recall that softmax requires a global maximum for numerical stability:
softmax(xi) =
exi−m
P
j exj−m ,
m = max
j
xj
The trick: we can update the running maximum and normalization factor as we process new blocks,
without ever materializing the full row.
Online Softmax Update Rule
Given a running state (mold, ℓold, Oold) and a new block of scores snew:
1. mnew = max(mold, max(snew))
2. ℓnew = emold−mnew · ℓold + P _jes_new,j−m_new
3. Onew =
1
ℓnew

em_old−m_new · ℓ_old · O_old + es_new−m_new · V _new

This is mathematically equivalent to computing softmax over all blocks at once.
1.6.3
The Flash Attention Algorithm
Flash Attention Forward Pass – Block Tiling
Setup: SRAM size M. Block sizes Br = ⌈M/(4d)⌉, Bc = min(⌈M/(4d)⌉, d).
1. Divide Q into Tr = ⌈n/Br⌉blocks Q1, . . . , QTr
2. Divide K, V into Tc = ⌈n/Bc⌉blocks K1, . . . , KTc
3. Initialize output O ∈Rn×d, running max m ∈Rn, running sum ℓ∈Rn (all in HBM)
4. Outer loop over j = 1, . . . , Tc:
(a) Load Kj, Vj from HBM to SRAM
(b) Inner loop over i = 1, . . . , Tr:
i. Load Qi, Oi, mi, ℓi from HBM to SRAM
ii. Compute Sij = QiKT
j /
√
d (stays in SRAM)
iii. Apply online softmax update to get new mi, ℓi, Oi
iv. Write Oi, mi, ℓi back to HBM
5. Return O
Key: Sij (the attention tile) is computed and discarded in SRAM. It is never written to HBM.
Flash Attention Complexity
Standard Attention
Flash Attention
Memory (HBM)
O(n2)
O(n)
HBM reads/writes
O(n2d)
O(n2d/M)
FLOPs
O(n2d)
O(n2d) (same)
Speedup
1×
2–4×
68


<!-- page 69 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
In the forward pass, the total FLOPs remain O(n2d) – identical to standard attention. Flash
Attention gains speed entirely by slashing slow HBM traffic, not by reducing arithmetic. (The
backward pass actually performs more FLOPs due to recomputation, but the wall-clock time is
still lower because the saved memory bandwidth dominates.)
1.6.4
Flash Attention 2 – Better Parallelism
Flash Attention 2 [82] made three key improvements:
1. Reduced non-matmul FLOPs: The original FA had unnecessary rescaling operations in
the inner loop. FA2 restructures the loop to minimize these. On A100, Tensor Core matrix
multiplications outpace scalar operations by roughly 16×, so even a small fraction of non-matmul
work in the inner loop becomes the latency bottleneck.
2. Better parallelism across sequence dimension: FA1 parallelized over batch and heads
only. FA2 also parallelizes over the query sequence dimension, enabling better GPU utilization
for long sequences with small batch sizes.
3. Causal masking optimization: For autoregressive (causal) attention, roughly half the blocks
are fully masked. FA2 skips these blocks entirely, giving ∼2× speedup for causal attention
vs. bidirectional.
1.6.5
Flash Attention 3 – Hopper Architecture
Flash Attention 3 [83] is designed specifically for H100 and exploits three Hopper-specific features:
• TMA (Tensor Memory Accelerator): H100 has a dedicated hardware unit for asynchronous
bulk data movement between HBM and SRAM. FA3 uses TMA to overlap data loading with
computation, hiding memory latency.
• Warp-specialization: FA3 assigns different warps to different roles (producer warps load
data via TMA; consumer warps compute MMA). This is a software pipelining technique that
keeps both the memory system and Tensor Cores busy simultaneously.
• FP8 support: H100 supports FP8 (E4M3/E5M2) Tensor Core operations at 2× the throughput
of BF16. FA3 supports FP8 attention with per-block quantization to maintain accuracy.
FA3 achieves up to 75% of H100 theoretical peak for FP16 attention, compared to ∼35% for
FA2.
1.6.6
Flash Attention 4 – Blackwell Architecture
Flash Attention 4 [84] targets NVIDIA’s Blackwell GPUs (B200/GB200), which double Tensor
Core throughput to 2.25 PFLOP/s (BF16) while non-matmul units (exponential, shared memory
bandwidth) scale at a slower rate. This asymmetric hardware scaling means that the bottleneck
shifts: on Blackwell, attention is limited not by matmul but by the softmax exponentials and shared
memory traffic surrounding them.
FA4 addresses this with four key techniques:
• Fully asynchronous MMA pipelines: Blackwell’s MMA instructions are fully asynchronous
(unlike Hopper’s wgmma which still blocked on completion). FA4 redesigns the pipeline to
overlap MMA, TMA loads, and softmax rescaling across larger tile sizes, keeping all hardware
units saturated.
• Software-emulated exponential: Instead of calling the hardware ex2 unit (which is the
throughput bottleneck), FA4 emulates ex using polynomial approximations executed on the
much faster Tensor Cores themselves. This trades extra matmul instructions for exponential-unit
stalls.
69


<!-- page 70 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Conditional softmax rescaling: Standard FlashAttention rescales the running max every
tile. FA4 skips the rescaling when the new tile’s max does not exceed the running max (common
in practice), saving both register shuffles and synchronization barriers.
• Tensor Memory + 2-CTA MMA mode (backward pass): The backward pass uses
Blackwell’s Tensor Memory (a per-SM scratchpad larger than shared memory) and a 2-CTA
cooperative mode that fuses dQ accumulation across two thread-block clusters, halving shared
memory round-trips.
FA4 Implementation: CuTe-DSL
FA4 is the first FlashAttention version written in CuTe-DSL, a Python-embedded domain-specific
language for GPU kernels (part of CUTLASS 4.x). CuTe-DSL compiles 20–30× faster than C++
CUTLASS templates while retaining full control over register allocation and pipeline scheduling.
This dramatically lowers the iteration time for kernel development.
Results.
On B200 with BF16 head-dim 128 (causal, seq-len 8K):
• 1613 TFLOP/s – 71% of Blackwell peak utilization
• 1.3× faster than cuDNN 9.13 (NVIDIA’s proprietary fused kernel)
• 2.7× faster than Triton on the same hardware
Hardware–Software Co-evolution
The FlashAttention series illustrates a key principle: each GPU generation shifts the bottleneck,
demanding new algorithmic ideas rather than just re-compilation. A80 →memory bandwidth
limited (FA1/FA2: tiling + recomputation). H100 →data movement limited (FA3: TMA + warp-
specialization). B200 →non-matmul compute limited (FA4: software-emulated exp + conditional
rescaling). Understanding where the hardware bottleneck lies is the prerequisite for writing efficient
kernels.
1.7
Pretraining: Best Practices
Pretraining is the most expensive phase of LLM development—consuming millions of GPU-hours
and requiring careful orchestration of data, compute, and hyperparameters. This section distills key
lessons from Llama-3 [25], Chinchilla [85], and GPT-4 [23].
1.7.1
Training Objective
All modern decoder-only LLMs use causal language modeling (CLM):
LCLM = −1
T
T
X
t=1
log Pθ(xt | x<t)
This simple objective—with enough data and scale—produces emergent capabilities (in-context
learning, reasoning, instruction following) without explicit supervision [21].
1.7.2
Data Pipeline
Pretraining Data Recipe
• Scale: 1–15 trillion tokens for frontier models (Llama-3: 15T tokens)
• Sources: Web crawl (80%), code (10%), books/papers (5%), curated (5%)
• Deduplication: MinHash + exact substring dedup reduces memorization [86]
70

