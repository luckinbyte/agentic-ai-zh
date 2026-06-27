<!-- page 39 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Use templates for structure: Encode task semantics via special tokens rather than
natural language instructions. E.g., <|tool_call|> is more reliable than “Now I will call a
tool:”.
• Tool/function calling: Define dedicated tokens like <|function|>, <|result|> to create
unambiguous boundaries between reasoning and action.
• Consistent handling in RL: During PPO/GRPO, ensure the reference model and pol-
icy model use identical tokenization and special token handling—mismatches corrupt KL
computation.
• EOS handling: During generation, ensure EOS is included in the action space. If the
model cannot emit EOS, responses grow unbounded (common RL failure mode).
1.3
The Transformer Architecture
The Transformer [6] is the foundation of all modern LLMs. Understanding its components is essential
for grasping every optimization and training method in this guide.
1.3.1
High-Level Structure
A decoder-only transformer processes tokens sequentially through embedding, repeated atten-
tion+FFN blocks, and a final projection to vocabulary logits.
Figure 1.3 shows the complete
architecture.
1.3.2
The Original Encoder-Decoder Transformer
The Transformer was originally introduced [6] as an encoder-decoder architecture for sequence-
to-sequence tasks (machine translation, summarization). While modern LLMs predominantly use
decoder-only variants (GPT-style), understanding the full architecture is essential because cross-
attention and masked self-attention — both originating here — remain fundamental building blocks.
Encoder.
The encoder processes the entire input sequence bidirectionally — each token attends
to all other tokens (no causal mask). This produces a rich contextual representation Henc ∈Rn×d
where each position encodes information about the full input:
• Input: Token embeddings + sinusoidal positional encodings
• Each layer: Multi-Head Self-Attention →Add & Norm →FFN →Add & Norm
• No causal mask: Position i attends to all positions 1, . . . , n
• Output: Contextual representations of the full input sequence
Decoder — Masked Multi-Head Self-Attention.
The decoder generates output tokens one at
a time (autoregressively). To prevent the model from “seeing the future,” the self-attention in the
decoder uses a causal mask:
MaskedAttn(Q, K, V ) = softmax
 
QKT
√dk
+ M
!
V
(1.1)
where the mask M is:
Mij =
(
0
if i ≥j (can attend)
−∞
if i < j (future token — blocked)
39


<!-- page 40 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Figure 1.3: Decoder-only Transformer block (GPT-style, Pre-Norm variant). Each sub-layer (attention,
FFN) is preceded by LayerNorm and followed by a residual addition: x + SubLayer(LN(x)). This Pre-Norm
ordering (used by Llama, GPT-3, Mistral) stabilizes training without warmup, unlike the original Post-Norm
(which applies LayerNorm after the addition). L identical blocks are stacked, followed by a final LayerNorm
and linear projection to vocabulary logits.
Why Masking Matters
During training, the decoder processes the entire target sequence in parallel (teacher forcing), but
each position must only attend to previous positions to maintain the autoregressive property. The
mask ensures that generating token t uses only information from tokens 1, . . . , t−1. At inference,
tokens are generated one-by-one so the mask is implicit — but during training it enables parallel
computation while preserving causality.
Decoder — Cross-Attention.
After masked self-attention, each decoder layer applies cross-
attention where the decoder attends to the encoder’s output representations. This is the mechanism
by which the decoder “reads” the input:
CrossAttn(Qdec, Kenc, Venc) = softmax
 
QdecKT
enc
√dk
!
Venc
(1.2)
• Queries come from the decoder’s previous sublayer (the masked self-attention output)
• Keys and Values come from the encoder’s final output Henc
• No mask is applied — every decoder position can attend to every encoder position
• This allows the decoder to dynamically focus on different parts of the input at each generation
step (e.g., attending to “cat” when translating to “gato” in English→Spanish)
40


<!-- page 41 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Figure 1.4: The original Transformer architecture (Vaswani et al., 2017). The encoder (left) processes
the full input with bidirectional self-attention. The decoder (right) generates tokens autoregressively using
masked self-attention and cross-attention to encoder representations. Dashed boxes indicate the repeated
layer block (×N); gray lines show residual connections bypassing each sub-layer. Note: the original work uses
Post-Norm (LayerNorm applied after the residual addition: LN(x + SubLayer(x))), unlike modern LLMs
which use Pre-Norm.
Full Decoder Layer.
Each decoder layer contains three sublayers (vs. two in the encoder):
1. Masked Multi-Head Self-Attention + Residual + LayerNorm
2. Multi-Head Cross-Attention (to encoder output) + Residual + LayerNorm
3. Feed-Forward Network + Residual + LayerNorm
From Encoder-Decoder to Decoder-Only.
Modern LLMs (GPT, Llama, Qwen) use only the
decoder, removing both the encoder and cross-attention layers entirely. The key insight: for generative
language modeling, a single causal (masked) self-attention stack is sufficient — the model learns to
encode context and generate continuations in a single pass. This simplifies architecture, training,
and inference while scaling more effectively. Encoder-decoder models (T5, BART) remain relevant
for tasks with distinct input/output structure (translation, summarization), and cross-attention
reappears in multimodal models where vision encoders provide keys/values to language decoders.
41


<!-- page 42 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.3.3
Decoder-Only vs Encoder-Decoder
Modern LLMs almost exclusively use decoder-only architectures, but understanding the trade-offs
with encoder-decoder designs clarifies why.
Architecture
Examples
Use Case
Decoder-only
GPT-4 [23], Llama [25], Mis-
tral [26], Qwen [32]
Autoregressive generation; dominant for
chat/reasoning
Encoder-decoder
T5 [29], BART [33], Flan-
T5 [34]
Seq2seq (translation, summarization);
less common now
Encoder-only
BERT [27], RoBERTa [35]
Classification/embeddings; not for gen-
eration
Why Decoder-Only Won
Decoder-only models are simpler (one model, one loss), scale better (all parameters contribute
to generation), and support unified training (pretraining = next-token prediction = fine-tuning
objective). Encoder-decoder models waste capacity on the encoder for pure generation tasks.
1.3.4
Embeddings: From Discrete Tokens to Continuous Space
Before any attention or computation happens, the transformer must convert discrete token IDs into
continuous vectors that neural networks can process. This is the role of the embedding layer.
What is an Embedding?
An embedding is a learned dense vector representation of a discrete
symbol. Instead of representing the word “king” as a one-hot vector of size |V| = 128,000 (mostly
zeros), we represent it as a compact vector in Rd (e.g., d = 4096) that captures its meaning.
The key insight: similar concepts get nearby vectors. In a well-trained embedding space:
• “king” and “queen” are close (both royalty)
• “king” and “bicycle” are far apart (unrelated)
• Vector arithmetic captures relationships:
⃗
king −
⃗
man +
⃗
woman ≈
⃗
queen
The Embedding Table.
In practice, the embedding layer is simply a matrix E ∈R|V|×d where
row i stores the embedding vector for token i:
embed(xt) = E[xt] ∈Rd
(1.3)
For a sequence of token IDs [x1, x2, . . . , xn], embedding is a simple table lookup (indexing opera-
tion):
H0 = [E[x1]; E[x2]; . . . ; E[xn]] ∈Rn×d
Embedding Table in Transformers
• Size: |V| × d. For Llama-3: 128,256 × 4,096 = 525M parameters (6.5% of 8B model).
• Initialization: Random (Xavier/normal), then learned via backpropagation.
• Weight tying: Many models share the embedding matrix with the output projection head:
Whead = ET . This saves parameters and creates a symmetric encode-decode structure.
• Input: Token ID (integer) →Output: Dense vector in Rd.
• Gradient flow: During training, only the rows corresponding to tokens in the current batch
receive gradient updates (sparse update).
42


<!-- page 43 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Figure 1.5: Embedding space visualization (2D projection): semantically similar words cluster together.
The embedding table learns these positions during pretraining, capturing meaning purely from co-occurrence
patterns in text.
Why Embeddings Work
The embedding table is learned end-to-end with the rest of the model. Because the model is
trained to predict the next token, it must learn representations where tokens that appear in similar
contexts get similar vectors. This is the distributional hypothesis: “you shall know a word by
the company it keeps” [36]. The embedding layer compresses this statistical structure into dense
geometry.
The Anisotropy Problem.
A critical issue arises when using pretrained embeddings (e.g., from
BERT or GPT-2) for downstream tasks like retrieval (RAG) or bootstrapping recommender systems:
the learned representations are highly anisotropic—they occupy a narrow cone in the embedding
space rather than being uniformly distributed across all directions [37].
Why this matters for applications:
• RAG / Retrieval: If all embeddings have cosine similarity > 0.7 regardless of content,
retrieval rankings become nearly random—the system cannot distinguish relevant from irrelevant
passages.
• Recommender systems: Using pretrained LLM embeddings to represent items/users only
works if the geometry preserves meaningful similarity structure.
• Clustering: Anisotropic embeddings collapse clusters, making it impossible to discover natural
groupings.
Resolution: Whitening.
A simple and effective fix is whitening [38]—a linear transformation
that makes the embedding distribution isotropic (zero mean, identity covariance):
˜h = D−1/2UT (h −µ)
(1.4)
where µ is the mean embedding, and UDUT is the eigendecomposition of the covariance matrix
Σ = 1
N
P
i(hi −µ)(hi −µ)T .
43


<!-- page 44 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Figure 1.6: Isotropy vs. anisotropy in embedding spaces. Left: isotropic embeddings spread uniformly,
making cosine similarity a reliable measure of semantic relatedness. Right: anisotropic embeddings (as found
in BERT) cluster in a narrow cone, causing all pairs to have high cosine similarity regardless of semantic
content. Whitening transforms the space to restore isotropy.
Whitening in Practice
• What it does: Rotates and scales the embedding space so all directions have equal variance
(unit covariance).
• Effect: Cosine similarity becomes meaningful—semantically similar pairs score high, dissim-
ilar pairs score low.
• Bonus: Can simultaneously reduce dimensionality by keeping only the top-k eigenvectors
(similar to PCA), making retrieval faster.
• Cost: Requires computing the covariance matrix over a representative corpus (one-time,
O(N · d2)). The transform itself is a simple matrix multiply at inference.
• Alternative approaches: Contrastive fine-tuning (SimCSE), flow-based normalization, or
training with isotropy-promoting regularizers.
1.3.5
Self-Attention Mechanism
Self-attention is the core operation that allows each token to attend to every other token in the
sequence, computing a weighted combination based on relevance.
Scaled Dot-Product Attention
Given input sequence X ∈Rn×d, we compute:
Q = XWQ,
K = XWK,
V = XWV
(WQ, WK, WV ∈Rd×dk)
Attention(Q, K, V ) = softmax
 
QKT
√dk
+ M
!
V
where M is the causal mask (for autoregressive models): Mij = 0 if i ≥j, else −∞.
Intuition: Each token “attends” to all previous tokens, computing a weighted average of their
values based on query-key similarity.
Computational Complexity.
The naive attention computation has quadratic cost in sequence
length:
• Time: O(n2 · d) — computing QKT requires n2 dot products, each of dimension dk.
44


<!-- page 45 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Memory: O(n2) — the full attention matrix must be materialized to apply softmax.
For a 128K-token context with d = 4096, the attention matrix alone is 128K × 128K = 16.4 billion
entries (64 GB in FP32). This quadratic scaling is the fundamental bottleneck for long-context
LLMs.
Table 1.4: Attention cost scaling: why naive implementation is prohibitive for long sequences.
Seq Length
Attention Ops
Matrix Size
Practical Impact
2K
4M
16 MB
Fast; fits in SRAM
8K
64M
256 MB
Manageable with FlashAttention
32K
1B
4 GB
Requires memory-efficient kernels
128K
16B
64 GB
Exceeds single GPU HBM
1M
1T
4 TB
Impossible without sub-quadratic methods
Approaches to Taming Attention Cost.
Several families of solutions address this quadratic
bottleneck:
1. Exact attention with IO-awareness (FlashAttention [7]): Does not reduce computational
complexity but eliminates the need to materialize the n × n matrix in HBM by computing
attention in tiles that fit in SRAM. Crucially, FlashAttention is orthogonal to the sparse
patterns below—it is an execution engine, not an attention pattern. Production systems
routinely combine FlashAttention with sliding windows or block-sparse masks, getting both IO
efficiency and reduced FLOPs. We cover the algorithm in detail in Section 1.6.
2. Sliding window / local attention: Each token only attends to the w nearest tokens (e.g.,
w = 4096). Cost becomes O(n · w)—linear in n. Used by Mistral [26] (window = 4096) and
Longformer [39]. Trades global context for efficiency; works well because most attention is local
in practice. In modern stacks, the sliding-window mask is executed inside a FlashAttention
kernel.
3. Sparse attention patterns: Combine local windows with periodic global tokens (e.g., every
512th token attends to all). BigBird [40] and LongT5 [41] use this. Preserves some long-range
connectivity at O(n√n) cost. Again, FlashAttention serves as the underlying kernel for the
non-zero attention blocks.
4. Linear attention / state-space models: Replace softmax(QKT )V with ϕ(Q)(ϕ(K)T V )
using associativity, or reformulate as a recurrence (Mamba [42], RWKV [43]). Theoretically
O(n · d2) total. Unlike approaches 2–3 above, these are architectural replacements that alter
model expressiveness—softmax-free attention is fundamentally less expressive, and empirically
these models still lag behind transformers on tasks requiring precise long-range retrieval or
complex reasoning.
5. KV cache compression: At inference, compress or evict old KV pairs to bound mem-
ory.
Techniques include: H2O [44] (heavy-hitter oracle—keep only high-attention keys),
StreamingLLM [45] (keep initial “attention sink” tokens + recent window), and quantized KV
caches [46].
FlashAttention + Sparse Patterns = Best of Both Worlds
A common misconception is that FlashAttention is an alternative to sparse attention. It is not—it
is an IO optimization for the attention kernel that composes freely with any attention mask.
Modern production systems (e.g., Mistral, DeepSeek) use FlashAttention as the execution engine
underneath a sliding-window or block-sparse mask. This gives you both reduced FLOPs (from
sparsity) and optimal memory access patterns (from tiling). RingAttention [47] extends this
further to multi-device settings, distributing the tiled computation across GPUs along the sequence
45


<!-- page 46 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
dimension.
Linear attention and state-space models (Mamba, RWKV) are a genuinely different architectural
choice—they sacrifice the full pairwise interaction for O(n) compute. While theoretically elegant,
they have not matched transformer quality on knowledge-intensive or long-range reasoning tasks,
and frontier labs continue to use exact attention (with FlashAttention + sparsity) as the backbone.
1.3.6
Multi-Head Attention
Rather than computing a single attention function, multi-head attention runs several attention
operations in parallel, each learning to focus on different aspects of the input (syntax, semantics,
position, etc.).
Multi-Head Attention
Instead of one attention function with d-dimensional keys/values, use H parallel heads with
dimension dk = d/H:
MultiHead(X) = Concat(head1, . . . , headH)WO
Each head can learn different attention patterns (e.g., one head for syntax, another for semantics,
another for positional proximity).
Grouped Query Attention (GQA): Llama-3 [25] uses fewer K,V heads than Q heads (e.g., 8
KV heads shared across 32 Q heads). This reduces KV cache size by 4× with minimal quality loss.
1.3.7
Positional Encodings
Transformers are permutation-equivariant by construction — without positional information, the
model cannot distinguish “the cat sat on the mat” from “mat the on sat cat the”. Positional encodings
inject sequence-order signal so that attention can reason about token distance and direction.
Table 1.5: Positional encoding methods in modern LLMs.
Method
Used By
Key Idea
Sinusoidal
Original Transformer
Fixed sin / cos at different frequencies.
Not learned.
Learned Absolute
GPT-2 [31], BERT [27]
Learned embedding per position. Lim-
ited to training length.
RoPE (Rotary)
Llama [25], Qwen [32], Mis-
tral [26]
Rotate
Q,K
vectors
by
position-
dependent angle.
Extrapolates via
NTK-aware scaling.
ALiBi
BLOOM [48], MPT [49]
No position embedding; add linear bias
−m|i −j| to attention scores. Simple,
extrapolates well.
Sinusoidal (Fixed) Positional Encoding.
Introduced in the original Transformer [6], this
method uses fixed sinusoidal functions at geometrically-spaced frequencies:
PE(pos, 2i) = sin

pos
100002i/d

,
PE(pos, 2i+1) = cos

pos
100002i/d

where pos is the token position, i is the dimension index, and d is the model dimension.
Motivation: Each frequency encodes position at a different scale (analogous to binary counting).
The authors hypothesised that the model could learn to attend to relative positions because PE(pos+k)
can be expressed as a linear function of PE(pos).
Pros: Zero learned parameters; deterministic; theoretically supports arbitrary lengths.
Cons: In practice, does not extrapolate well beyond training lengths; the model must learn to
decode relative position from absolute signals indirectly; largely superseded.
46


<!-- page 47 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Learned Absolute Positional Embedding.
Used by GPT-2 [31] and BERT [27]: a learnable
embedding matrix Epos ∈RLmax×d is added to token embeddings:
h(pos)
0
= TokenEmbed(xpos) + Epos[pos]
Motivation: Let the model learn whatever positional representation is optimal for the task,
rather than imposing a fixed structure.
Pros: Maximum flexibility; simple implementation; often outperforms sinusoidal for short se-
quences.
Cons: Hard-coded maximum length Lmax; no generalisation beyond it; embeddings near the end
of Lmax are under-trained; adds Lmax × d parameters.
Rotary Position Embedding (RoPE).
RoPE [50] encodes position by rotating query and key
vectors in 2D subspaces:
RoPE(xm, m) =









x(1)
m
x(2)
m
...
x(d−1)
m
x(d)
m









⊙








cos mθ1
cos mθ1
...
cos mθd/2
cos mθd/2








+









−x(2)
m
x(1)
m
...
−x(d)
m
x(d−1)
m









⊙








sin mθ1
sin mθ1
...
sin mθd/2
sin mθd/2








where θi = 10000−2i/d and m is the position index. The key property is that the dot product between
rotated queries and keys depends only on relative position:
⟨RoPE(qm, m), RoPE(kn, n)⟩= f(qm, kn, m −n)
Motivation: Achieve relative position encoding without explicit bias terms, while maintaining
compatibility with linear attention and KV-caching.
Pros: Naturally relative; no extra parameters; compatible with efficient inference; can be extended
to longer contexts via NTK-aware scaling [51] or YaRN (adjusting θ base or interpolating frequencies).
Cons: Slightly more compute per attention operation (rotation + interleaving); extrapolation
requires explicit scaling strategies; rotation in 2D subspaces imposes structure that may not be
optimal for all tasks.
RoPE Length Extension
To extend a RoPE model trained at L to context length L′ > L:
• Position interpolation: Scale positions by L/L′ so all positions fit in [0, L]. Simple but
compresses resolution.
• NTK-aware scaling: Increase the θ base (e.g. 10000 →10000 · (L′/L)d/(d−2)), effectively
stretching high-frequency components while preserving low-frequency ones.
• YaRN [51]: Combines NTK scaling with an attention temperature correction t = 0.1 ln(s)+1
to compensate for increased entropy at longer distances.
ALiBi (Attention with Linear Biases).
ALiBi [52] takes a radically different approach: no
positional embedding at all. Instead, a static linear penalty is subtracted from attention scores:
Attention(Q, K, V ) = softmax
 
QKT
√dk
−m ·
|i −j|

i,j
!
V
where m is a head-specific slope (set geometrically: mh = 2−8h/H for head h of H total). The bias
−m|i −j| creates a soft local attention window whose width varies by head.
47


<!-- page 48 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Motivation: Position should bias attention toward nearby tokens (recency prior) without
interfering with the embedding space. By operating purely in attention-score space, ALiBi avoids
polluting token representations with positional signal.
Pros: Excellent length extrapolation (trained at 1k, works at 8k+); zero parameters; trivial to
implement; head-specific slopes give multi-scale locality.
Cons: Less expressive for tasks requiring precise long-range positional reasoning (e.g. “what was
the 5th word?”); the linear decay is a strong inductive bias that may not suit all domains; largely
overtaken by RoPE in recent models due to RoPE’s better short-context performance.
Table 1.6: Positional encoding comparison: practical trade-offs.
Sinusoidal
Learned Abs.
RoPE
ALiBi
Extra parameters
None
Lmax × d
None
None
Position type
Absolute
Absolute
Relative
Relative
(im-
plicit)
Length extrapolation
Poor
None
Good (w/ scal-
ing)
Excellent
Compute overhead
Negligible
Negligible
Small
Negligible
Dominant era
2017–19
2018–20
2022–present
2022–23
Scaling to Extremely Long Contexts (100K–1M+ Tokens).
Modern frontier models
(Claude [53] with 200K–1M context, Gemini 1.5 [54] at 1M+, GPT-4 [23] at 128K) require po-
sitional encodings that remain faithful far beyond training lengths. The dominant solutions today:
1. RoPE with frequency scaling: The standard approach for extending RoPE beyond training
length. Rather than retraining, the base frequency θ is rescaled:
θ′
i = θi ·
Ltarget
Ltrain
2i/d
Variants include:
• Linear scaling (Position Interpolation) [55]: Simply divide position indices by a factor s.
Cheap but degrades quality at high extension ratios.
• NTK-aware scaling [51]: Scale the base frequency θ = 10000 →10000 · sd/(d−2).
Preserves high-frequency (local) information while extending low-frequency (global) range.
• YaRN [51] (Yet another RoPE extensioN): Combines NTK scaling with an attention
temperature correction and fine-tuning on a small long-context corpus. Used by Llama-3
to extend from 8K training to 128K deployment.
• Dynamic NTK [51]: Adjusts the scaling factor on-the-fly based on actual sequence
length at inference. No fixed extension ratio needed—the model adapts as context grows.
2. Continued pretraining on long data: Even with RoPE scaling, models benefit from a
short continued pretraining phase (1–5B tokens) on long documents. This teaches the model
to actually use distant context, not just tolerate it positionally. Llama-3.1 used a progressive
schedule: 8K →64K →128K.
3. Ring Attention / Blockwise Parallel [47]: For sequences exceeding single-GPU memory
(1M+ tokens), Ring Attention distributes the sequence across GPUs in a ring topology. Each
GPU holds a block and passes KV blocks around the ring, computing local attention tiles. This
enables linear memory scaling with GPU count while preserving exact attention.
4. Hybrid architectures: Some systems combine a local sliding window (e.g., 4K) for most
layers with full attention at select layers (e.g., every 4th layer). This provides O(n · w) cost for
most computation while maintaining global information flow.
48


<!-- page 49 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Long Context ̸= Long Context Usage
A model with 1M context length does not necessarily use all 1M tokens effectively. The “lost in
the middle” phenomenon [56] shows that models tend to focus on the beginning and end of long
contexts, underutilizing information in the middle. Effective long-context utilization requires both
positional encoding support and training on tasks that reward long-range retrieval.
1.3.8
Feed-Forward Network (MLP)
Each transformer block contains an MLP applied independently to each position:
FFN(x) = W2 · σ(W1x + b1) + b2
where W1 ∈Rd×4d, W2 ∈R4d×d. Modern LLMs use:
• SwiGLU activation: FFN(x) = W2(Swish(W1x) ⊙W3x) — used by Llama [25], Mistral [26].
Requires 3 weight matrices but gives better performance.
• Hidden dimension is typically 8/3 × d (rounded to multiples of 256 for Tensor Core efficiency).
FFN as Memory
Recent work [57] suggests the FFN layers act as a key-value memory: W1 rows are keys (patterns
to match), W2 columns are values (information to output). The FFN “retrieves” stored knowledge
based on the current hidden state.
1.3.9
Layer Normalization
Layer normalization stabilizes training by normalizing activations across the feature dimension. Its
placement relative to the attention/FFN sublayers significantly affects training dynamics.
How LayerNorm Works.
Given a hidden state vector x ∈Rd (a single token’s representation),
LayerNorm [58] computes:
LayerNorm(x) = γ ⊙
x −µ
√
σ2 + ϵ + β
(1.5)
where:
• µ = 1
d
Pd
i=1 xi (mean across the d feature dimensions)
• σ2 = 1
d
Pd
i=1(xi −µ)2 (variance across features)
• γ, β ∈Rd are learned scale and shift parameters (per-dimension)
• ϵ ≈10−5 prevents division by zero
Key distinction from BatchNorm: LayerNorm normalizes across the feature dimension of a
single example, not across the batch. This makes it independent of batch size and works identically
at training and inference.
RMSNorm — The Modern Simplification.
RMSNorm [59] drops the mean-centering step,
normalizing only by the root-mean-square:
RMSNorm(x) = γ ⊙
x
RMS(x),
RMS(x) =
v
u
u
t1
d
d
X
i=1
x2
i
(1.6)
No β (shift) parameter and no mean subtraction — just scale. This saves one reduction operation
per token and is ∼5–10% faster on GPUs while achieving equivalent model quality. All modern
LLMs (Llama, Mistral, Qwen) use RMSNorm.
49


<!-- page 50 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Pre-LN vs Post-LN
• Post-LN (original Transformer): h + LayerNorm(Attn(h)).
Requires careful warmup;
training can be unstable.
• Pre-LN (GPT-2+, all modern LLMs): h+Attn(LayerNorm(h)). Stabilizes training; enables
higher learning rates.
• RMSNorm (Llama [25], Mistral [26]): Simplified LayerNorm without mean-centering:
RMSNorm(x) = x/RMS(x) · γ. Slightly faster, same quality.
Why Normalization Matters for Deep Networks
Without normalization, activations tend to grow or shrink exponentially through layers (explod-
ing/vanishing activations). A 128-layer transformer without LayerNorm would see magnitudes
vary by 1030× between the first and last layer. Normalization constrains each layer’s output to
a predictable range, enabling stable gradient flow and allowing the optimizer to use consistent
learning rates throughout the network.
1.3.10
Model Size Reference
The following table summarizes key architectural parameters for widely-used open-weight models
(latest versions as of 2025), providing a quick reference for understanding scale and design choices.
Table 1.7: Architecture parameters for popular open-weight LLMs (2024–2025 generation).
Model
Params
Layers
d
Heads
KV
Heads
Context
Llama-3.1 8B [25]
8B
32
4096
32
8
128K
Llama-3.1 405B [25]
405B
126
16384
128
8
128K
Llama-4 Maverick [60]
400B (17B
active)
48
5120
40
8
1M
Mistral Large 2 [61]
123B
88
12288
96
8
128K
Qwen-2.5 72B [32]
72B
80
8192
64
8
128K
DeepSeek-V3 [62]
671B (37B
active)
61
7168
128
MLA
128K
Note: Models marked with “active” parameters use Mixture-of-Experts (MoE) architecture—
total parameters indicate model capacity, while active parameters reflect per-token compute cost.
DeepSeek-V3 uses Multi-head Latent Attention (MLA) instead of standard GQA, compressing KV
into a low-rank latent space.
1.3.11
Attention Pathologies
While the attention mechanism is powerful, it exhibits systematic failure modes that practitioners
must understand—especially when scaling to long contexts or interpreting model behaviour.
Attention Sink
The phenomenon.
Xiao et al. [63] discovered that transformer models allocate disproportionately
high attention scores to the first token in the sequence—regardless of its semantic content. Even
when the first token is a meaningless <BOS> marker, attention heads across all layers consistently
attend to it, sometimes with 20–50% of total attention mass.
Why it happens.
Softmax attention must produce a valid probability distribution (P
j αj = 1).
When no key is particularly relevant to a query, the model needs a “dump” location for unused
attention mass. During training, the first token becomes this default sink because it is always present
and positionally predictable. It functions as a no-op attention target—the model has learned to route
irrelevant attention there rather than distributing it unpredictably.
50


<!-- page 51 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
αsink =
exp(q⊤k0/
√
d)
P
j exp(q⊤kj/
√
d)
≫1
n
(even when k0 is semantically irrelevant)
Consequences.
• Streaming inference failure: When using sliding-window KV caches, evicting the first token
causes perplexity to spike catastrophically—the model loses its attention sink.
• Misleading interpretability: Naive attention visualizations suggest the first token is “impor-
tant” when it is merely a mathematical artefact.
• Context window waste: The sink token occupies a KV cache slot without carrying useful
information.
Solutions.
• StreamingLLM [63]: Always keep the first k tokens (“attention sinks”) in the KV cache
alongside the recent sliding window. Enables infinite-length generation with bounded memory.
• Sink tokens by design: Some models (e.g., Mistral) prepend dedicated sink tokens during
training that are explicitly meant to absorb residual attention.
• Softmax alternatives: Replace softmax with ReLU attention or sigmoid gating, where zero
attention is representable without requiring a dump target.
Attention Dilution
The phenomenon.
As sequence length n grows, each query must distribute its attention budget
across more keys. The average attention weight per token decreases as O(1/n), making it progressively
harder for the model to concentrate on the few truly relevant positions—a problem known as attention
dilution or attention diffusion [56].
The “Lost in the Middle” effect.
Liu et al. [56] showed that LLMs exhibit a U-shaped retrieval
curve: information placed at the beginning or end of long contexts is retrieved reliably, but information
in the middle is often ignored. This is a direct consequence of attention dilution compounded with
positional biases from RoPE/ALiBi:
Why it happens.
• Softmax saturation: With many keys, the softmax temperature effectively decreases, making
the distribution more uniform (entropic).
• Positional decay: RoPE’s relative positional encoding introduces a natural decay with
distance, suppressing attention to middle positions that are far from both start and end.
• Training distribution: Models trained on shorter sequences develop attention patterns biased
toward recent context.
Mitigation strategies.
• Explicit retrieval: Place relevant context at the beginning or end of the prompt; use RAG to
avoid relying on middle positions.
• Long-context training: Train on long documents with varied placement of key informa-
tion [64].
51


<!-- page 52 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Hierarchical attention: Architectures like Mamba [65] or RWKV that avoid the O(n2)
attention bottleneck entirely.
• Landmark tokens: Insert retrievable markers in the context that act as “signposts” for
attention.
• Temperature scaling: Some implementations scale the attention logits by log n to counteract
dilution in long sequences.
Other Attention Phenomena
Table 1.8: Additional attention patterns observed in large transformers.
Pattern
Description
Implication
Attention heads specialization
Different
heads
learn
dis-
tinct roles:
syntax heads,
co-reference heads, positional
heads [66]
Not all heads are equally important;
many can be pruned
Induction heads
Heads
that
implement
[A][B]...[A] →[B] copying [67]
Critical for in-context learning; emerge
in 2-layer+ models
Attention collapse
In deep networks, attention
distributions can converge (all
heads attend same positions)
Hurts expressivity; addressed by atten-
tion diversity losses
Retrieval heads
Specific heads specialize in re-
trieving factual information
from context [68]
Explains why pruning certain heads
causes hallucination spikes
1.3.12
Visualizing Attention for Explainability
Attention weights provide a window into model reasoning—but must be interpreted carefully.
Attention Visualization Methods
Raw attention maps.
The simplest approach: plot the n×n attention matrix A = softmax(QK⊤/
√
d)
as a heatmap for each head and layer. Tools like BertViz [69] render interactive multi-head visualiza-
tions.
Attention rollout.
Raw attention at a single layer is misleading because information flows through
residual connections across all layers. Abnar and Zuidema [70] propose attention rollout: multiply
attention matrices across layers to approximate the total information flow from input to output:
R(l) = A(l) · R(l−1),
R(0) = I
where A(l) is the (averaged across heads) attention matrix at layer l, adjusted to include the residual
connection: A(l) = 0.5 · A(l)
raw + 0.5 · I.
Gradient-weighted attention.
Combine attention weights with gradient information to identify
which attended tokens actually influence the output [71]:
Relevance(i) = αi ·

∂y
∂hi

This addresses the criticism that high attention ̸= high influence (a token can receive high attention
but be processed through a near-zero-weight path).
52


<!-- page 53 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Attention Is Not Explanation
Jain and Wallace [72] showed that attention weights often do not correlate with gradient-based
feature importance and that adversarial attention distributions can produce identical outputs.
Use attention visualization as a hypothesis generator, not as a faithful explanation. For causal
attribution, prefer gradient-based methods, probing, or mechanistic interpretability.
Mechanistic Interpretability with Sparse Autoencoders (SAEs)
The interpretability problem.
Individual neurons in transformer MLPs and residual streams are
typically polysemantic—a single neuron activates for multiple unrelated concepts (e.g., “the colour
blue AND academic citations AND the word ‘the”’). This makes direct neuron-level interpretation
unreliable.
Sparse Autoencoders.
Cunningham et al. [73] and Bricken et al. [74] demonstrated that training
a sparse autoencoder (SAE) on model activations can decompose polysemantic representations into
monosemantic features—interpretable directions that each correspond to a single concept:
h = Wdec · ReLU(Wenc · x + benc) + bdec
where Wenc ∈Rm×d with m ≫d (overcomplete basis), and the ReLU + sparsity penalty ensures
only a few features activate per input.
Key findings from SAE interpretability:
• Features are monosemantic: each encodes a single human-interpretable concept (“code in
Python,” “mentions of the Golden Gate Bridge,” “first-person narrative”) [74].
• Features are steerable: clamping a feature’s activation high/low directly controls model behaviour
(e.g., forcing the “Golden Gate Bridge” feature on makes the model mention it in every
response) [75].
• Features compose: complex behaviours emerge from combinations of simple features.
• SAEs scale: Templeton et al. [75] trained SAEs with up to 34M features on Claude 3 Sonnet,
finding interpretable features for safety-relevant concepts (deception, sycophancy, dangerous
requests).
SAE Training Recipe
1. Collect activations from a specific model layer across a large corpus.
2. Train a sparse autoencoder with L1 penalty on the hidden layer: L = ∥x −ˆx∥2
2 + λ∥z∥1.
3. The learned encoder directions (Wenc rows) are candidate features.
4. Validate: for each feature, find max-activating examples and check semantic coherence.
5. Optionally: measure feature absorption and dead features to assess SAE quality.
Natural Language Autoencoders (Anthropic, 2026)
While SAEs decompose activations into interpretable vectors, their features still require human
inspection of max-activating examples to understand. Anthropic’s Natural Language Autoencoders
(NLAEs) [76] take a fundamentally different approach: they replace the sparse bottleneck with
natural language descriptions, making interpretability automatic.
53

