<!-- page 103 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
Limitations of Model-Level Detection
These methods detect uncertainty, not incorrectness. A model can be confidently wrong (low
entropy, consistent responses—but factually false). For reliable detection, combine with retrieval-
based verification (RAG) or external fact-checking tools.
1.17
LLM Safety and Responsible AI
Safety is not an afterthought—it is an integral part of the LLM training pipeline. This section covers
the key dimensions of LLM safety and the mechanisms used to enforce responsible behavior.
1.17.1
Threat Taxonomy
Table 1.21: LLM safety threat categories.
Category
Description and Examples
Harmful content
Generating toxic, violent, or illegal instructions (bioweapons,
CSAM)
Bias and discrimination
Perpetuating stereotypes; unfair treatment across demograph-
ics [153]
Privacy violations
Leaking PII from training data; memorization attacks [154]
Jailbreaking
Adversarial prompts that bypass safety guardrails [155]
Misinformation
Generating convincing but false claims (hallucination at scale)
Dual-use
Legitimate capabilities (coding, chemistry) weaponized for harm
1.17.2
Safety Training Pipeline
Figure 1.14: Safety is applied at every stage: data filtering in pretraining, refusal examples in SFT, safety-
specific reward models in RLHF, and iterative red-teaming.
1.17.3
Key Safety Mechanisms
Safety Techniques
• Data filtering: Remove toxic, biased, and PII-containing text from pretraining corpora
• Safety SFT: Train on examples of appropriate refusals (“I can’t help with that because. . . ”)
• Constitutional AI [129]: Self-critique using principles; model revises its own outputs
against a constitution of rules
• Safety reward model: Separate RM trained on safety-annotated pairs; combined with
helpfulness RM during RLHF via weighted sum
• Guardrails: Input/output classifiers that block harmful requests/responses at serving time
• Red teaming [156]: Systematic adversarial evaluation to find failure modes before deploy-
ment
103


<!-- page 104 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
1.17.4
The Helpfulness–Safety Tradeoff
Balancing Helpfulness and Safety
Over-optimizing for safety creates an over-refusal problem: the model declines benign requests
(e.g., refusing to discuss historical violence in an educational context). The goal is a Pareto-optimal
policy that is maximally helpful within safety constraints:
max
θ
E[Rhelpful]
subject to
E[Rsafety] ≥τ
In practice, this is implemented as a weighted reward: R = αRhelpful + (1 −α)Rsafety with careful
tuning of α (typically 0.6–0.8). Meta’s Llama-3 reports using distinct safety and helpfulness reward
models with margin-based weighting [25].
1.17.5
Evaluation
• Safety benchmarks: ToxiGen, RealToxicityPrompts, BBQ (bias), CrowS-Pairs
• Jailbreak robustness: GCG attacks [155], multi-turn jailbreaks, encoded prompts
• Over-refusal rate: Measure false-positive refusals on benign prompts (target <5%)
• Red team evaluations: Human adversarial testing with domain experts (biosecurity, cyber-
security)
Safety Is Never Complete
No combination of techniques provides absolute safety. New attack vectors are discovered con-
tinuously (multi-modal jailbreaks, fine-tuning attacks that remove safety training, many-shot
prompting). Safety requires ongoing monitoring, rapid response to new threats, and defense-in-
depth (multiple independent layers).
104
