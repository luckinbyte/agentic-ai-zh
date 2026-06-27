<!-- page 87 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
At runtime, looking up the mask is an O(1) table access—adding negligible latency to each decoding
step.
Key libraries.
• Outlines [115]: Compiles JSON schemas and regexes into interleaved FSM-guided generation.
Supports any model with a logits interface.
• lm-format-enforcer1: Similar FSM approach with a focus on integration with serving
frameworks (vLLM, TGI).
• Guidance2 (Microsoft): Interleaves constrained generation with control flow (loops, conditions),
enabling complex structured outputs beyond flat schemas.
• XGrammar [116]: Pushdown-automaton-based engine supporting full context-free grammars
(not just regular languages), used in MLC-LLM and vLLM for grammar-mode decoding.
Trade-offs.
Constrained decoding guarantees syntactic validity—no post-hoc parsing failures, no
retries. However:
• Semantic quality: Forcing structure can degrade content quality if the model’s probability
mass for the “correct” answer lies outside the grammar. In practice this is rare for well-trained
models on well-designed schemas.
• Compilation cost: The FSM index must be built per schema. For complex schemas this can
take 1–5 s, but it is amortized over all requests using that schema.
• Grammar coverage: Regex/FSM handles JSON, YAML, SQL fragments, and most structured
formats. Full CFGs (via XGrammar or LALR parsers) cover languages like Python or XML.
When to Use Constrained Decoding
Use constrained decoding whenever the consumer of the model’s output is a program rather
than a human. Tool-calling agents, API backends, and data-extraction pipelines all benefit from
guaranteed valid structure. For free-form prose or creative text, unconstrained sampling remains
appropriate.
1.13
Prompt Engineering
Prompt engineering is the discipline of designing inputs to LLMs that reliably elicit desired behaviour—
without changing model weights. While fine-tuning modifies the model, prompt engineering exploits
the model’s existing capabilities through careful framing, examples, and structure. It is the fastest,
cheapest, and most accessible lever for improving LLM outputs, and remains essential even when
using fine-tuned models.
1.13.1
In-Context Learning (ICL)
In-context learning [21] is the remarkable ability of large language models to learn tasks at inference
time purely from examples provided in the prompt—with no gradient updates. The model implicitly
infers the task from the pattern of input–output pairs and generalizes to new inputs.
Why In-Context Learning Works
• Implicit Bayesian inference: The model has seen millions of tasks during pretraining.
The prompt examples locate the relevant task in the model’s learned distribution [117].
1https://github.com/noamgat/lm-format-enforcer
2https://github.com/guidance-ai/guidance
87


<!-- page 88 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
• Induction heads: Specific attention heads learn to copy patterns (“A is to B as C is to ”),
enabling in-context generalization [67].
• Task vectors: ICL creates implicit task representations in the residual stream that steer
generation toward the demonstrated format and content [118].
Scaling behaviour.
ICL emerges primarily in models above ∼1B parameters and improves log-
linearly with model scale [21]. Smaller models can memorize examples but struggle to generalize to
novel inputs within the same context window.
1.13.2
Zero-Shot Prompting
Zero-shot prompting provides no examples—only a task description or instruction. The model must
rely entirely on its pretrained knowledge and instruction-tuning to produce the correct format and
content.
Zero-Shot Classification
Classify
the
following
movie
review as POSITIVE or NEGATIVE.
Review: "The
cinematography
was
breathtaking
but the plot
felt
rushed and
predictable."
Sentiment:
When zero-shot works well:
• Tasks the model has seen extensively during pretraining/SFT (translation, summarization,
sentiment)
• Well-specified instructions with unambiguous output format
• Instruction-tuned models (e.g., ChatGPT, Claude, Llama-3-Instruct) significantly outperform
base models at zero-shot tasks [9]
When zero-shot fails:
Novel formats, domain-specific labeling schemes, or ambiguous tasks where
the model cannot infer your exact requirements from the instruction alone.
1.13.3
Few-Shot Prompting
Few-shot prompting [21] provides k input–output examples (“shots”) before the actual query. This
is the most common form of in-context learning and remains one of the most effective prompting
strategies.
Few-Shot Named Entity Recognition
Extract
named
entities
from the text. Format: [ENTITY ]( TYPE)
Text: "Apple
released
the iPhone 15 in Cupertino."
Entities: [Apple ](ORG), [iPhone
15]( PRODUCT), [Cupertino ](LOC)
Text: "Elon Musk
announced
Tesla ’s new
factory in Berlin."
Entities: [Elon Musk ](PER), [Tesla ](ORG), [Berlin ](LOC)
Text: "OpenAI
partnered
with
Microsoft to deploy GPT -4."
Entities:
88


<!-- page 89 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Key design principles for few-shot examples:
1. Diversity: Cover the range of expected inputs (different lengths, edge cases, categories).
2. Ordering: Place harder or more representative examples last (recency bias) [119].
3. Label balance: If classifying, include examples from all classes to avoid majority-class bias.
4. Format consistency: Every example must follow the exact same structure. The model mimics
the pattern.
5. Relevance: Use examples semantically similar to the target query for best results [120].
How many shots?
Performance typically improves from 0 to 4–8 examples, then plateaus. Beyond
∼20 examples, gains are marginal and you risk filling the context window. Min et al. [121] showed
that the format and label space of examples matter more than label correctness—even random labels
help (though correct labels help more).
1.13.4
Instruction-Following Prompts
Instruction-tuned models respond best to clear, structured instructions. The key insight: treat the
prompt as a specification, not a suggestion.
Anatomy of an Effective Instruction Prompt
1. Role/Persona: Define who the model is (“You are a senior data scientist...”)
2. Task: What to do, stated clearly and unambiguously
3. Context: Background information the model needs
4. Constraints: Length limits, tone, what to avoid, output format
5. Examples (optional): Show the desired output format
6. Input: The actual data to process
Instruction Prompt with Constraints
Role: You are a medical
literature
reviewer.
Task: Summarize
the
following
research
abstract
for a
general
audience.
Constraints:
- Maximum 3 sentences
- No jargon (explain
any
technical
terms)
- Include
the key
finding
and its
clinical
implication
- Do NOT
speculate
beyond
what the
abstract
states
Abstract: [...]
System prompts vs. user prompts.
Modern chat APIs separate the system prompt (persistent
instructions, role definition) from the user message (per-turn input). System prompts are processed
with higher attention priority in most models and provide a natural place for role definitions,
constraints, and output format specifications [23].
1.13.5
Structured Output Prompts (JSON/XML)
For programmatic use, the most critical prompting technique is enforcing structured output—
particularly JSON.
89


<!-- page 90 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
JSON Output Prompt
Extract
the
following
information
from the
customer
email.
Respond
ONLY with
valid JSON , no other
text.
Schema:
{
"intent": "refund|complaint|question|praise",
"urgency": "low|medium|high",
" product_mentioned ": "string or null",
"summary": "one
sentence
summary"
}
Email: [...]
Techniques for reliable structured output:
• Schema-first: Show the exact JSON schema before the input. The model treats it as a
template.
• Constrained decoding: Use grammar-based sampling (e.g., Outlines [115], Guidance) to
guarantee syntactically valid JSON at the token level.
• XML tags: For nested or multi-part outputs, XML tags (e.g., <thinking>...</thinking>)
provide unambiguous delimiters that models follow reliably.
• Pydantic/TypeScript types: Providing type definitions helps models understand field
constraints (OpenAI’s function calling uses JSON Schema internally).
JSON in Prompts — Common Pitfalls
• Models may add markdown fences (“‘json ...
“‘) — instruct explicitly to output raw
JSON.
• Nested objects and arrays increase hallucination risk — flatten schemas where possible.
• Enum fields (fixed choices) are much more reliable than free-text fields.
• Always validate outputs programmatically; no prompt guarantees 100% compliance without
constrained decoding.
JSON Prompting: Structuring the Input.
A distinct but complementary technique is JSON
prompting—formatting the prompt itself as JSON rather than natural language. This exploits
the model’s extensive pre-training on structured data (APIs, configs, code) to improve instruction
adherence, reduce ambiguity, and enable deterministic parsing of multi-field requests.
JSON Prompting with System Prompt
=== SYSTEM ===
You are a senior
code
reviewer. Analyze
code for bugs ,
security
issues , and style
violations. Always
respond
in the JSON
schema
provided.
=== USER (JSON
prompt) ===
{
"task": "code_review",
"language": "python",
" severity_filter ": "high",
"code": "def login(user , pw):\n
query = ...",
" output_schema": {
"issues": [{
"line": "int",
90


<!-- page 91 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
"severity": "critical|high|medium|low",
"category": "security|bug|style|performance",
"description": "string",
"fix": "string"
}],
"overall_risk": "critical|high|medium|low"
}
}
Why JSON prompting works:
• Unambiguous field boundaries: No confusion about where one instruction ends and
another begins.
• Typed constraints: Fields like "severity_filter":
"high" are clearer than “only show
high severity issues.”
• Schema-as-contract: Including output_schema in the input mirrors API design patterns
the model has seen extensively during pre-training.
• System prompt still essential: The system message provides role, tone, and behavioral
constraints that don’t fit naturally in a JSON payload.
1.13.6
Chain-of-Thought (CoT) Prompting
Chain-of-thought prompting [122] asks the model to produce intermediate reasoning steps before
giving a final answer. This simple technique dramatically improves performance on tasks requiring
multi-step reasoning: arithmetic, logic, commonsense inference, and code generation.
Why CoT works:
• Serializes computation: Transformers have fixed depth but variable-length generation. CoT
converts parallel (hard) problems into sequential (easy) steps, effectively increasing the model’s
computational budget.
• Reduces compounding errors: Each step is a simpler sub-problem with lower per-step error
rate.
• Exposes intermediate state: Makes reasoning auditable and debuggable.
Chain-of-Thought Variants
Method
Description
Zero-shot CoT [123]
Append “Let’s think step by step” to any prompt
Few-shot CoT [122]
Provide examples with explicit reasoning chains
Self-Consistency [124]
Sample N CoT paths; majority-vote the final answer
Tree of Thoughts [125]
Explore multiple reasoning branches with backtracking
Plan-and-Solve [126]
First plan the steps; then execute each step
ReAct [127]
Interleave Reasoning and Acting (tool use)
Zero-Shot Chain-of-Thought
Q: A store has 45 apples. They sell 3/5 of them in the
morning
and 1/3 of the
remaining in the
afternoon.
How many
apples are left?
Let’s think
step by step.
A: Morning
sales: 45 * 3/5 = 27 apples
sold.
Remaining
after
morning: 45 - 27 = 18.
91


<!-- page 92 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Afternoon
sales: 18 * 1/3 = 6 apples
sold.
Remaining: 18 - 6 = 12 apples.
Self-Consistency.
Wang et al. [124] showed that sampling multiple chain-of-thought reasoning
paths and taking a majority vote over final answers significantly outperforms single-path CoT. The
intuition: correct reasoning paths tend to converge on the same answer, while errors are typically
idiosyncratic. This trades compute (generating N samples) for accuracy—practical when latency is
less important than correctness.
When CoT hurts.
CoT is not universally beneficial. For simple tasks (single-step classification,
retrieval, formatting), CoT adds unnecessary tokens, increases latency, and can even introduce errors
through overthinking. Use CoT selectively for tasks where you expect multi-step reasoning to be
required.
1.13.7
Advanced Prompting Techniques
Retrieval-Augmented Generation (RAG).
Rather than relying solely on the model’s parametric
memory, RAG [128] retrieves relevant documents and includes them in the prompt:
Context (retrieved): [document
chunks]
Question: [user
query]
Answer
based
ONLY on the
provided
context.
This grounds the model’s responses in verifiable sources and dramatically reduces hallucinations
for knowledge-intensive tasks.
Prompt Chaining and Decomposition.
Complex tasks benefit from being broken into a pipeline
of simpler prompts, where the output of one becomes the input to the next:
1. Extract key facts from document
2. Reason over extracted facts
3. Format final answer
Each step can use a different prompt template, model, or temperature setting. This is more
controllable than a single monolithic prompt and enables targeted debugging.
Constitutional AI / Self-Critique.
Bai et al. [129] introduce prompts that ask the model to
critique and revise its own output against a set of principles:
[Generate
initial
response]
Critique: Does this
response
violate
any of the
following
principles? [list
principles]
Revision: Rewrite
the
response
addressing
the
critique.
Meta-Prompting and Prompt Optimization.
Rather than hand-crafting prompts, recent work
automates prompt design:
• APE [130]: Uses an LLM to generate and score candidate prompts automatically.
• DSPy [131]: Compiles declarative task descriptions into optimized prompt pipelines with
learned few-shot examples.
• OPRO [132]: Treats prompt optimization as an optimization problem, using an LLM as the
optimizer.
92


<!-- page 93 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Attentive Reasoning Queries (ARQ).
ARQ [133] addresses a fundamental weakness of standard
prompting: as context length grows, models increasingly “lose” critical information in the middle of
the prompt (the lost-in-the-middle effect). ARQ mitigates this by decomposing a complex query into
multiple focused sub-queries, each designed to direct the model’s attention to a specific part of the
context:
1. Query decomposition: Break the user question into atomic sub-questions that each target a
narrow aspect.
2. Attentive retrieval: For each sub-query, retrieve or highlight only the relevant context
slice—forcing the model to attend to it.
3. Aggregation: Combine sub-answers into a coherent final response.
This is particularly effective for long-document QA, multi-hop reasoning over large retrieval sets,
and agentic tasks where the context window contains many tool outputs. ARQ can be seen as a
structured form of chain-of-thought that explicitly manages where the model looks, not just how it
reasons.
1.13.8
Best Practices: Crafting Effective Prompts
Based on empirical findings across the literature and practitioner experience, the following principles
reliably improve prompt quality:
The Prompt Engineering Checklist
1. Be specific and unambiguous: Replace “summarize this” with “summarize in 2–3 bullet
points, each under 20 words, focusing on actionable findings.”
2. Show, don’t tell: One good example is worth 100 words of instruction. When in doubt,
add a few-shot example.
3. Define the output format explicitly: Specify JSON schema, bullet points, table format,
or exact delimiters. Never leave format to interpretation.
4. Use
delimiters
for
input
data:
Wrap user inputs in clear delimiters (""",
<input>...</input>, –-) to separate instructions from data.
5. Assign a role: “You are a [domain expert] who [specific behaviour]” primes relevant
knowledge and tone.
6. Specify what NOT to do: Negative constraints (“do not explain your reasoning”, “never
output more than 5 items”) are often more effective than positive ones.
7. Add chain-of-thought for reasoning tasks: Append “Think step by step” or provide
worked examples for math, logic, or multi-hop questions.
8. Control temperature appropriately: Use T ≈0 for factual/deterministic tasks; T ≈0.7–
1.0 for creative/diverse outputs.
9. Iterate empirically: Treat prompts as code—version them, A/B test them, and measure
performance on representative eval sets.
10. Leverage recency bias: Place the most critical instructions and examples at the end of
the prompt (closest to the generation point).
The Prompt Engineering Mindset
Think of prompt engineering as programming in natural language. The model is a powerful but
literal interpreter—it will do exactly what you ask, interpreted in the most likely way given its
training distribution. Common principles from software engineering apply:
93


<!-- page 94 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Table 1.16: Common prompting failure modes and solutions.
Failure Mode
Symptom
Solution
Instruction amnesia
Model ignores constraints in
long prompts
Move constraints to end; repeat key
rules; use system prompt
Format drift
Output starts correct but de-
grades over long generations
Use constrained decoding; break into
shorter chained prompts
Sycophancy
Model agrees with incorrect
premises in the prompt
Add “challenge assumptions if incor-
rect”; use system-level instruction
Hallucinated details
Model invents facts not in pro-
vided context
Add “if unknown, say I don’t know”;
use RAG with source attribution
Refusal over-triggering
Model refuses benign requests
due to safety training
Rephrase to clarify legitimate intent;
provide explicit context for why the re-
quest is appropriate
• DRY (Don’t Repeat Yourself): Unless fighting attention decay in long contexts
• Separation of concerns: Different prompt sections for role, constraints, examples, and
input
• Test-driven development: Define expected outputs before writing the prompt
• Version control: Track prompt iterations and their eval scores
• Modularity: Build reusable prompt templates; parameterize variable parts
When prompting fails to achieve the desired quality after systematic iteration, that is the signal
to move to fine-tuning (SFT) or reinforcement learning (RLHF/DPO).
1.14
Model Compression Methods
Model compression reduces model size and inference cost while preserving quality. Three main
approaches: quantization (reduce precision), pruning (remove parameters), and distillation (train a
smaller model to mimic a larger one).
1.14.1
Quantization
Quantization reduces model size and inference cost by representing weights (and optionally activations)
in lower-precision formats. The core trade-off is compression ratio versus quality degradation.
Quantization Overview
Quantization reduces the numerical precision of model weights (and optionally activations) from
FP32/BF16 to lower-bit formats:
xq = round
x −z
s

,
xdequant = s · xq + z
where s is the scale factor and z is the zero-point.
When to Quantize
• Inference serving: Always quantize. W4A16 (4-bit weights, BF16 activations) is the sweet
spot — 2× memory savings, <1% quality loss for 70B+ models.
• Training: FP8 on H100 gives 2× throughput with minimal quality loss. BF16 is still the
default for smaller models.
• Edge deployment: GGUF Q4_K_M for local inference on consumer hardware.
94


<!-- page 95 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Table 1.17: Quantization methods for LLMs.
Method
Bits
Type
Key Idea
GPTQ [134]
4-bit
PTQ, weight-only
Layer-wise quantization minimiz-
ing ∥WX −ˆWX∥2 via optimal
brain surgeon.
AWQ [135]
4-bit
PTQ, weight-only
Protects salient weights (those
with large activations).
1% of
weights carry 99% importance.
GGUF [136]
2–8 bit
PTQ, weight-only
CPU-optimized
format
(llama.cpp). Per-block quantiza-
tion with multiple types.
FP8 (E4M3)
8-bit
Training + inference
Native
H100
support.
2×
throughput vs BF16.
SmoothQuant [137]
W8A8
PTQ, weight+act.
Smooths activation outliers into
weights before quantization. En-
ables INT8 GEMM.
QAT [138]
4-bit
QAT
Trains with simulated quantiza-
tion. Highest quality but expen-
sive.
AQLM [139]
2-bit
PTQ, additive codes
Extreme compression via learned
additive quantization codebooks.
• RLHF: Quantize the frozen models (reference, reward model) to INT8/FP8. Keep the
policy in BF16 for training precision.
1.14.2
Pruning
Why Prune?
Modern LLMs contain billions of parameters, yet empirical studies consistently show
that a large fraction of these weights contribute minimally to model outputs. Pruning exploits this
over-parameterization: by removing redundant weights, we reduce memory footprint (enabling
deployment on smaller GPUs or edge devices), inference latency (fewer multiply-accumulate
operations per forward pass), and serving cost (higher throughput per dollar). Unlike quantization,
which reduces the precision of all weights uniformly, pruning selectively eliminates weights—enabling
multiplicative savings when combined with quantization (e.g., a 50% sparse, 4-bit model uses 4× less
memory than the dense BF16 baseline). The challenge is achieving high sparsity without degrading
generation quality, which has driven the development of principled one-shot methods that require no
retraining.
Pruning Methods
• Unstructured pruning: Zero out individual weights below a threshold. High sparsity
(50–90%) possible. Requires sparse GEMM kernels (2:4 on A100/H100).
• Structured pruning: Remove entire attention heads, layers, or FFN neurons. Directly
reduces FLOPS without specialized kernels.
• SparseGPT [140]: One-shot pruning using approximate inverse Hessian. 50% unstructured
sparsity with minimal quality loss on 175B models.
• Wanda [141]: Prune by |w| × ∥x∥(weight magnitude times input activation norm). No
calibration data needed. Competitive with SparseGPT.
NVIDIA 2:4 Structured Sparsity
A100/H100 Tensor Cores natively support 2:4 sparsity: out of every 4 elements, at most 2 are
non-zero. This gives exactly 2× speedup on supported operations with hardware acceleration. The
constraint: you must achieve exactly 50% sparsity in this specific pattern, which limits flexibility
compared to arbitrary sparsity.
95


<!-- page 96 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.14.3
Knowledge Distillation
Knowledge distillation [142] transfers the learned behaviour of a large teacher model into a smaller,
cheaper student model. The core idea is that the teacher’s output distribution over tokens carries far
richer signal than ground-truth hard labels alone — revealing inter-class similarities, calibration, and
uncertainty that the student can exploit.
Temperature-Scaled Softmax.
To expose the “dark knowledge” in the teacher’s logits we soften
the distribution with a temperature T > 1:
p(T)
i
=
exp(zi/T)
P
j exp(zj/T)
At high temperature the probability mass spreads across more tokens, making near-miss alternatives
visible. During training the same temperature is applied to the student; at inference the student uses
T = 1.
General Distillation Loss.
Ldistill = α T 2 · KL
 P (T)
teacher ∥P (T)
student
 + (1 −α) · LCE(y, P (1)
student)
The T 2 factor compensates for the reduced gradient magnitude of softened distributions. Typical
values: T ∈[2, 20], α ∈[0.5, 0.9] (more weight on KL when teacher quality is high).
Table 1.18: Knowledge distillation paradigms for LLMs.
Paradigm
Mechanism
Pros
Cons
Offline / White-box
Teacher
logits
pre-
computed;
student
trains on full distribu-
tions
Full distribution sig-
nal; one-time teacher
cost
Stale
data;
storage
heavy
Online / Co-training
Teacher generates on-
the-fly; student sees
fresh logits
Adapts
to
student
weaknesses
2× compute; complex in-
fra
Black-box (API)
Only teacher text out-
puts available (no log-
its)
Works with propri-
etary models
Loses dark knowledge;
SFT-like
Self-distillation
Model distills into a
smaller version of it-
self
No separate teacher
needed
Teacher = student fam-
ily; ceiling
Offline (White-Box) Distillation.
The teacher’s full logit vector (or top-k logits for storage
efficiency) is recorded for each training token. The student minimises the KL divergence against these
stored distributions. This is the most data-efficient paradigm when teacher access is unrestricted.
Motivation: Decouple teacher inference from student training — run the teacher once on high-end
hardware, then train many students cheaply.
Pros: Deterministic, reproducible; teacher cost is amortised; full distributional signal.
Cons: Requires storing |V |-dimensional vectors per token (mitigated by top-k pruning); teacher
cannot adapt to student failures.
Online (Co-Training) Distillation.
Teacher and student are run jointly: the teacher generates
logits for the student’s current training batch.
Motivation: Let the teacher focus on inputs where the student currently struggles (curriculum-
like).
96


<!-- page 97 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Pros: Freshness; can use student-generated inputs for on-policy distillation.
Cons: Double the GPU cost; synchronisation complexity; harder to scale.
Black-Box (API) Distillation.
When only text outputs are available (e.g. distilling from a
proprietary API), the student is trained via SFT on the teacher’s generations, optionally augmented
with chain-of-thought traces.
Motivation: Practical reality — most frontier models do not expose logits.
Pros: Simple pipeline; works with any model behind an API.
Cons: No soft-label signal; prone to hallucination amplification; effectively supervised fine-tuning.
Self-Distillation.
A model distils from a larger version within the same architecture family
(e.g. Llama-3 70B →8B) or from its own checkpoints during training.
Motivation: Avoid training a separate teacher; leverage the model’s own capacity at different
scales.
Pros: Architecture compatibility; no external dependency.
Cons: Teacher ceiling equals model ceiling; cannot introduce genuinely new knowledge.
Dark Knowledge
Consider a language model predicting the next word after “The capital of France is”. Hard labels
say only “Paris” is correct. But the teacher’s soft distribution might assign 5% to “Lyon”, 2% to
“Marseille”, and near-zero to “banana” — telling the student which errors are reasonable, which
dramatically improves calibration and generalisation.
Practical Considerations for LLM Distillation.
• Sequence-level vs. token-level: Token-level KL is standard; sequence-level distillation
(minimising KL over full sequences) better captures long-range coherence but is harder to
optimise.
• Layer-wise hints: Matching intermediate representations (attention maps, hidden states)
provides additional learning signal — especially useful when student architecture differs.
• Data selection: Distillation data quality matters; curating diverse, hard examples yields
better students than random sampling.
• Student capacity: Diminishing returns below ∼10% of teacher parameters; at extreme
compression, architecture changes (e.g. MoE →dense) may be needed.
• Combining with quantization: Distillation + 4-bit quantization (e.g. QLoRA-distilled
models) achieves near-teacher quality at 20× compression.
97


<!-- page 98 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Compression Method Comparison – 70B Model
Method
Size
Speed
Quality
Use Case
BF16 (baseline)
140 GB
1×
100%
Training,
refer-
ence
FP8 (E4M3)
70 GB
2×
99.5%
H100 inference
INT8 (SmoothQuant)
70 GB
1.8×
99%
A100 inference
4-bit (AWQ)
35 GB
2.5×
97–98%
Serving at scale
2-bit (AQLM)
17.5 GB
3×
90–93%
Edge, experimen-
tal
Pruned 50% (2:4)
70 GB
1.8×
97%
Structured
speedup
Distilled 8B
16 GB
10×
80–85%
Mobile, edge
1.15
Speculative Decoding Methods
Speculative decoding [143] accelerates autoregressive generation by predicting multiple tokens simul-
taneously, then verifying them in a single forward pass of the target model. It produces identical
output distribution to standard decoding (no quality loss) while achieving 2–3× speedup.
1.15.1
Core Principle
Speculative Decoding Framework
1. A fast draft mechanism proposes k candidate tokens: ˆx1, . . . , ˆxk
2. The large target model runs a single forward pass on all k tokens (batched)
3. Verification: Accept tokens left-to-right while Ptarget(ˆxi) ≥ri·Pdraft(ˆxi) (where ri ∼U[0, 1])
4. On first rejection at position j:
resample xj from an adjusted distribution, discard
ˆxj+1, . . . , ˆxk
Key property: This acceptance/rejection scheme guarantees the final distribution equals Ptarget
exactly.
Speedup: If acceptance rate is α, expected tokens per step = 1−αk+1
1−α . At α = 0.8, k = 5: expected
3.4 tokens/step vs. 1 for standard decoding.
1.15.2
Methods Comparison
1.15.3
Medusa: Multi-Head Speculative Decoding
How Medusa Works
Medusa adds k additional “prediction heads” to the LLM (sharing the same backbone):
• Head 0 (original): predicts token at position t + 1 (standard next-token)
• Head 1: predicts token at position t + 2 (skipping one)
• Head i: predicts token at position t + i + 1
• All heads run in parallel during a single forward pass
• A tree-structured verification validates multiple candidate sequences simultaneously
Training: Fine-tune only the Medusa heads (backbone frozen). Cost: ∼1 epoch on representative
data.
98


<!-- page 99 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Table 1.19: Speculative decoding methods supported by modern inference engines.
Method
Draft Source
Speedup
Key Idea
Standard [143]
Small model (1–7B)
2–3×
Separate draft model generates can-
didates. Simple but requires loading
2 models.
Medusa [144]
Parallel LM heads
2–3×
Add k extra prediction heads to the
target model. Each predicts token
at position +1, +2, . . . , +k.
Eagle [145]
Feature-level
2.5–3.5×
Lightweight decoder generates draft
tokens from target model’s hidden
states.
Higher acceptance than
Medusa.
Eagle-2 [145]
Context-aware
3–4×
Dynamic draft tree with confidence-
based expansion. State-of-the-art ac-
ceptance rates.
N-gram Lookup
N-gram cache
1.5–2×
Match prompt n-grams against pre-
viously generated text.
Zero cost;
great for repetitive outputs.
Lookahead [146]
Jacobi iteration
2–2.5×
Parallel Jacobi decoding with n-
gram verification. No draft model;
uses target model itself.
Multi-token [147]
Modified arch.
2–3×
Train the model to natively predict
multiple tokens per step (Meta’s ap-
proach in Llama).
Advantage: No separate draft model; heads are tiny (one linear layer each). Memory overhead:
<1%.
1.15.4
Eagle: Feature-Level Drafting
Why Eagle Outperforms Medusa
Medusa’s heads predict independently at each position — they cannot condition on their own
previous predictions (token at t + 2 doesn’t know what was predicted at t + 1). Eagle fixes this
with a lightweight autoregressive decoder that operates on the target model’s hidden states:
1. Extract hidden states from the target model’s last layer
2. Feed into a small (1-layer) decoder that autoregressively generates draft tokens conditioned
on previous hidden states
3. This captures inter-token dependencies that Medusa misses
Result: Eagle achieves 85–95% acceptance rate vs. Medusa’s 60–80%.
1.15.5
N-gram Speculative Decoding
N-gram Lookup Method
The simplest speculative decoding — requires no additional model or training:
1. Maintain a cache of n-grams from the prompt and previously generated text
2. At each step, check if the current context’s last n −1 tokens match any cached n-gram
3. If yes: propose the continuation as draft tokens
99


<!-- page 100 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
4. Verify against target model as usual
Best for: Code generation (repetitive patterns), structured outputs (JSON/XML), and prompts
with repeated elements. Cost: Essentially zero.
100


<!-- page 101 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.15.6
Integration with vLLM
from vllm
import LLM , SamplingParams
# Standard
speculative
decoding (separate
draft
model)
llm = LLM(
model="meta -llama/Llama -3-70B",
tensor_parallel_size =4,
speculative_config ={
"model": "meta -llama/Llama -3-8B",
" num_speculative_tokens ": 5,
},
)
# N-gram
speculation (zero -cost , no draft
model
needed)
llm = LLM(
model="meta -llama/Llama -3-70B",
speculative_config ={
"method": "ngram",
" num_speculative_tokens ": 5,
" prompt_lookup_max ": 4,
# Match up to 4-grams
from
prompt
},
)
# EAGLE -style (feature -level draft , high
acceptance
rate)
llm = LLM(
model="meta -llama/Meta -Llama -3-8B-Instruct",
tensor_parallel_size =4,
speculative_config ={
"model": "yuhuili/EAGLE -LLaMA3 -Instruct -8B",
" num_speculative_tokens ": 2,
"method": "eagle",
" draft_tensor_parallel_size ": 1,
},
)
# MLP
speculator (IBM -style , lightweight
head)
llm = LLM(
model="meta -llama/Meta -Llama -3.1 -70B-Instruct",
tensor_parallel_size =4,
speculative_config ={
"model": "ibm -ai -platform/llama3 -70b-accelerator",
" draft_tensor_parallel_size ": 1,
},
)
When NOT to Use Speculative Decoding
• High batch sizes: At batch ≥64, generation is already compute-efficient. Speculation
adds overhead (draft generation + verification) that doesn’t pay off.
• Very different distributions: If draft model is too dissimilar to target, acceptance rate
drops below 50% and speculation is slower than standard decoding.
• Short outputs: For <20 token outputs, the setup cost of speculation exceeds savings.
• Rule of thumb: Speculation helps most for latency-sensitive, single-stream generation
(chatbots, interactive code completion).
101


<!-- page 102 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.16
Hallucination Detection
LLMs generate fluent text that may be factually incorrect—a phenomenon called hallucination [148].
This section covers basic detection methods at the model level (without external retrieval or multi-
agent verification).
1.16.1
Types of Hallucination
Hallucination Taxonomy
• Intrinsic: Contradicts the provided input/context (e.g., summary says the opposite of the
source)
• Extrinsic: Generates claims that cannot be verified from the input and are factually wrong
• Faithfulness: Output diverges from the instruction or specified constraints
1.16.2
Detection Methods (Model-Level)
Table 1.20: Basic hallucination detection methods that operate at the model level.
Method
Mechanism
Signal
Token-level entropy
High entropy at generation time indi-
cates uncertainty [149]
H(P(xt)) > τ
Sequence log-prob
Low average log-probability of the out-
put suggests confabulation
1
T
P
t log P(xt)
Consistency sampling
Generate N responses; low agreement
= likely hallucination [150]
Contradiction rate
Semantic entropy
Cluster meanings (not strings); high
semantic entropy = uncertain [151]
Cluster diversity
DoLA
Contrast logits between later vs. earlier
layers; amplifies factual knowledge [152]
Layer divergence
Semantic Entropy.
Kuhn et al. [151] observe that token-level entropy is unreliable (paraphrases
have different tokens but same meaning). Instead, they generate multiple responses, cluster them by
semantic equivalence (via NLI), and compute entropy over meaning clusters:
SE = −
X
c∈clusters
P(c) log P(c)
High SE means the model produces semantically different answers—a strong hallucination signal.
SelfCheckGPT.
Manakul et al. [150] detect hallucinations by checking self-consistency: generate
multiple responses and verify whether claims in the main response are supported by the alternatives.
If the model “disagrees with itself,” the claim is likely hallucinated. No external knowledge needed.
DoLA (Decoding by Contrasting Layers).
Chuang et al. [152] observe that factual knowledge
emerges in later transformer layers while earlier layers retain more generic/uncertain representations.
DoLA contrasts the logit distributions between a later (“mature”) layer and an earlier (“premature”)
layer at each decoding step:
DoLA(xt) = softmax
 log Plate(xt) −log Pearly(xt)

By amplifying the signal from factual knowledge encoded in deeper layers, DoLA reduces hallucinations
at inference time without any retraining—requiring only a single additional forward pass through the
contrasted layer. It is complementary to sampling-based methods and can be combined with them.
102

