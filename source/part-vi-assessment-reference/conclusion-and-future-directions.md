

<!-- page 573 -->
Chapter 29
Conclusion and Future Directions
29.1
Summary
This guide has traced the full arc from transformer foundations through reinforcement learning for
alignment to the construction of autonomous agentic systems. The key themes that emerge across
all chapters:
1. Alignment is a systems problem. It is not enough to have a good loss function. Production
RLHF requires managing 4+ models, distributing computation across hundreds of GPUs,
handling fault tolerance, and monitoring for reward hacking—all simultaneously.
2. There is no single best method. PPO remains the gold standard for maximum quality but
demands enormous engineering investment. DPO and its variants offer compelling trade-offs
for teams with limited infrastructure. GRPO bridges the gap for verifiable-reward domains.
The right choice depends on your data, compute budget, and quality bar.
3. Reasoning emerges from reward. DeepSeek-R1 proved that chain-of-thought, self-verification,
and backtracking can emerge from simple binary reward signals and group-relative optimization—
without explicit demonstrations of reasoning. Test-time compute scaling means smaller models
with more thinking can match larger models.
4. Standards unlock ecosystems. MCP reduces the tool integration problem from N × M to
N + M. A2A enables agents built by different teams to collaborate without shared internals.
These protocols are to agentic AI what HTTP was to the web—the enabling infrastructure for
an open ecosystem.
5. Agents are the natural next step. Once a model is aligned, the frontier shifts from “how
good is a single response?” to “can the model solve multi-step problems autonomously?” This
requires new training paradigms (agentic RL with environment rewards), new infrastructure
(harnesses, tool protocols, memory systems), and new evaluation methods (trajectory-level
benchmarks).
6. Evaluation drives everything. Without rigorous evaluation—from reward model validation
to agent task success rates, from contamination detection to LLM-as-Judge calibration—progress
is unmeasurable and regressions are invisible. The benchmarks you choose shape the systems
you build.
7. Simplicity scales. The most reliable production agents use the simplest architecture that
meets requirements—prompt chaining and routing before autonomous loops, single agents
before multi-agent swarms. Complexity should be earned through demonstrated need.
573


<!-- page 574 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
29.2
The Road Ahead: Open Challenges
29.2.1
Learning from Interaction
Current RLHF pipelines [9] treat alignment as a one-time training phase. The future points toward
continuous learning from deployment: agents that improve from every user interaction, tool
failure, and environment observation—without catastrophic forgetting [204] or reward drift. Key
open problems:
• Online learning with non-stationary reward distributions.
• Safe exploration in production [404] (avoiding harmful actions while learning).
• Efficient credit assignment over long agent trajectories (hundreds of tool calls).
29.2.2
Scalable Oversight
As agents become more capable, human oversight becomes the bottleneck. Current approaches
(RLHF [9], Constitutional AI [129]) rely on humans evaluating model outputs—but what happens
when model outputs exceed human understanding?
• Recursive reward modeling [175]: Use AI to help humans evaluate AI.
• Debate and amplification [405]: Two models argue; a human judges which argument is
more compelling.
• Process-based supervision [243]: Reward correct reasoning steps, not just final answers.
• Mechanistic interpretability [67]: Understand what the model is doing internally, not just
what it outputs.
29.2.3
World Models and Planning
Current agents are reactive—they observe and respond one step at a time. Future agents will need
internal world models [172] that enable lookahead planning:
• Predicting the consequences of actions before executing them.
• Tree search over possible action sequences (à la AlphaGo [19] and MuZero [171] but for
open-ended tasks).
• Learning environment dynamics from interaction traces.
29.2.4
Multi-Agent Ecosystems
The A2A protocol [372] and multi-agent frameworks hint at a future where hundreds of specialized
agents collaborate, negotiate, and delegate—forming an “economy of agents” [394]. Open challenges:
• Trust and verification between agents with different principals.
• Emergent cooperation vs. emergent deception in competitive settings [406].
• Market mechanisms for resource allocation (compute, tool access, priority).
• Governance: who is responsible when a chain of 10 agents produces a harmful outcome? [407]
574


<!-- page 575 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
29.2.5
Agent Security and Trust
Autonomous agents inherit every security vulnerability of the LLMs they are built on—plus new attack
surfaces created by tool access, multi-agent delegation, and persistent memory (Chapters 19–21).
Critical unsolved problems:
• Prompt injection at scale [408]: As agents consume untrusted content (web pages, emails,
API responses), indirect prompt injection becomes systemic. No robust defense exists today.
• Confused deputy attacks: An agent with legitimate credentials can be tricked into misusing
them on behalf of an attacker embedded in the data stream [335].
• Sandboxing without crippling: Least-privilege execution constrains what agents can do,
but overly restrictive sandboxes negate agentic value. Finding the right boundary is an open
design problem.
• Audit and attribution: When an agent chain spans multiple organizations (via A2A [372]),
tracing who authorized what action remains architecturally unsolved.
• Trust calibration: Agents must learn when not to trust—whether a tool response is authentic,
whether another agent’s claim is verified.
29.2.6
Evaluation Beyond Benchmarks
Chapter 14 showed that benchmarks shape the systems we build—yet current evaluation has critical
gaps:
• Real-world deployment metrics: Benchmarks like SWE-bench [266] and GAIA [362]
measure isolated tasks; production agents face ambiguous goals, shifting requirements, and
multi-turn recovery.
• Reward model validity: RLHF assumes reward models capture human preferences, but
reward hacking [409] and distributional shift undermine this assumption at scale.
• Cost-quality frontiers: Two agents may achieve the same accuracy, but one costs 10× more
tokens. Evaluation must become cost-aware.
• Safety under distribution shift: An agent safe in testing may behave unsafely on novel
inputs. Adversarial evaluation [156] and red-teaming at agentic scale remain immature.
29.2.7
Efficiency and Accessibility
Training a 70B model with RLHF costs 10K −−100K. Running autonomous agents costs 1 −−50
per complex task. For agentic AI to achieve broad impact:
• Distillation of agentic capabilities from large to small models [142, 410].
• More efficient RL algorithms (fewer samples, lower variance) [168].
• On-device agents that operate without cloud round-trips.
• Open-weight models that match proprietary quality for agentic tasks [15].
575


<!-- page 576 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
29.3
Further Reading
29.3.1
Foundational Papers
• Attention Is All You Need [6] — The transformer architecture.
• RLHF / InstructGPT [9] — The first large-scale RLHF deployment.
• PPO [168] — Proximal Policy Optimization.
• DPO [10] — Direct Preference Optimization.
• GRPO / DeepSeek-R1 [14, 15] — Group Relative Policy Optimization and emergent
reasoning.
• ReAct [127] — Reasoning + Acting framework for LLM agents.
• Toolformer [332] — Teaching LLMs to use tools.
• RAG [128] — Retrieval-Augmented Generation.
29.3.2
Systems and Scaling
• Megatron-LM [207] — Tensor and pipeline parallelism.
• DeepSpeed ZeRO [213] — Memory-efficient distributed training.
• vLLM [157] — PagedAttention for efficient LLM serving.
• Flash Attention [7] — IO-aware exact attention.
29.3.3
Agentic AI
• Building Effective Agents [342] — Design patterns and principles.
• Voyager [228] — Open-ended agent with skill library in Minecraft.
• SWE-bench [266] — Benchmark for autonomous software engineering.
• OSWorld [356] — Full computer-use benchmarks.
• GAIA [362] — General AI Assistants benchmark for real-world tasks.
• MemGPT [316] — OS-inspired memory management for unbounded context.
• Model Context Protocol [335] — Open standard for tool integration.
• Agent-to-Agent Protocol [372] — Inter-agent communication standard.
29.3.4
Alignment and Safety
• Constitutional AI [129] — Self-supervised alignment.
• Sleeper Agents [406] — Deceptive alignment concerns.
• Reflexion [224] — Learning from verbal self-reflection.
• Indirect Prompt Injection [408] — Security risks for LLM-integrated applications.
576


<!-- page 577 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
29.3.5
Online Resources
• HuggingFace TRL: https://github.com/huggingface/trl — Production RL library.
• LangGraph: https://github.com/langchain-ai/langgraph — Agent workflow graphs.
• OpenAI Agents SDK: https://github.com/openai/openai-agents-python — Official agent
framework.
• DeepSpeed-Chat: https://github.com/microsoft/DeepSpeedExamples — End-to-end RLHF
pipeline.
• DSPy: https://github.com/stanfordnlp/dspy — Declarative prompt optimization.
• AutoGen: https://github.com/microsoft/autogen — Multi-agent conversation framework.
“The best way to predict the future is to build it.”
— Alan Kay
577


<!-- page 578 -->
Bibliography
[1] Michael Wooldridge, Nicholas R. Jennings, and David Kinny. The Gaia Methodology for
Agent-Oriented Analysis and Design. Autonomous Agents and Multi-Agent Systems, 2000.
[2] Fabio Luigi Bellifemine, Giovanni Caire, and Dominic Greenwood. JADE: Developing Multi-
Agent Systems with JADE, 2007.
[3] Foundation for Intelligent Physical Agents. FIPA ACL Message Structure Specification, 2002.
URL http://www.fipa.org/specs/fipa00061/.
[4] Tim Berners-Lee, James Hendler, and Ora Lassila. The Semantic Web. Scientific American,
2001.
[5] Avigdor Gal, Ateret Anaby-Tavor, Alberto Trombetta, and Danilo Montesi. A Framework
for Modeling and Evaluating Automatic Semantic Reconciliation. In Proceedings of the 31st
International Conference on Very Large Data Bases (VLDB), 2005. URL https://link.
springer.com/chapter/10.1007/11896548_42.
[6] Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. Attention Is All You Need. In Advances
in Neural Information Processing Systems (NeurIPS), 2017. URL https://arxiv.org/abs/
1706.03762.
[7] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast
and Memory-Efficient Exact Attention with IO-Awareness. In Advances in Neural Information
Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2205.14135.
[8] Edward J. Hu, Yelong Shen, Phillip Wallis, et al. LoRA: Low-Rank Adaptation of Large
Language Models. arXiv Preprint arXiv:2106.09685, 2022.
[9] Long Ouyang, Jeffrey Wu, Xu Jiang, et al. Training Language Models to Follow Instructions
with Human Feedback. In Advances in Neural Information Processing Systems (NeurIPS),
2022. URL https://arxiv.org/abs/2203.02155.
[10] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon,
and Chelsea Finn.
Direct Preference Optimization: Your Language Model Is Secretly a
Reward Model. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL
https://arxiv.org/abs/2305.18290.
[11] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO:
Model Alignment as Prospect Theoretic Optimization. arXiv Preprint arXiv:2402.01306, 2024.
URL https://arxiv.org/abs/2402.01306.
[12] Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, et al. A General Theoretical Paradigm
to Understand Learning from Human Feedback. arXiv Preprint arXiv:2310.12036, 2024. URL
https://arxiv.org/abs/2310.12036.
[13] Jiwoo Hong, Noah Lee, and James Thorne. ORPO: Monolithic Preference Optimization
Without Reference Model. arXiv Preprint arXiv:2403.07691, 2024. URL https://arxiv.org/
abs/2403.07691.
578


<!-- page 579 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[14] Zhihong Shao, Peiyi Wang, Qihao Zhu, et al. DeepSeekMath: Pushing the Limits of Mathe-
matical Reasoning in Open Language Models. arXiv Preprint arXiv:2402.03300, 2024. URL
https://arxiv.org/abs/2402.03300.
[15] DeepSeek-AI, Daya Guo, Dejian Yang, et al. DeepSeek-R1: Incentivizing Reasoning Capability
in LLMs via Reinforcement Learning. arXiv Preprint arXiv:2501.12948, 2025. URL https:
//arxiv.org/abs/2501.12948.
[16] Murray Campbell, A. Joseph Hoane Jr., and Feng hsiung Hsu. Deep Blue. Artificial Intelligence,
2002.
[17] David Ferrucci, Eric Brown, Jennifer Chu-Carroll, et al. Building Watson: An Overview of the
DeepQA Project. AI Magazine, 2010.
[18] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet Classification with Deep
Convolutional Neural Networks. NeurIPS, 2012.
[19] David Silver, Aja Huang, Chris J. Maddison, et al. Mastering the Game of Go with Deep
Neural Networks and Tree Search. Nature, 2016. URL https://www.nature.com/articles/
nature16961.
[20] David Silver, Julian Schrittwieser, Karen Simonyan, et al. Mastering the Game of Go Without
Human Knowledge. Nature, 2017. URL https://www.nature.com/articles/nature24270.
[21] Tom Brown, Benjamin Mann, Nick Ryder, et al. Language Models Are Few-Shot Learners.
NeurIPS, 2020.
[22] John Jumper, Richard Evans, Alexander Pritzel, et al. Highly Accurate Protein Structure
Prediction with AlphaFold. Nature, 2021.
[23] OpenAI. GPT-4 Technical Report. arXiv Preprint arXiv:2303.08774, 2023.
[24] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural Machine Translation of Rare
Words with Subword Units. In Proceedings of the 54th Annual Meeting of the ACL, 2016. URL
https://arxiv.org/abs/1508.07909.
[25] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The Llama 3 Herd of Models.
arXiv Preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.
[26] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. Mistral 7B. arXiv Preprint
arXiv:2310.06825, 2023. URL https://arxiv.org/abs/2310.06825.
[27] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-Training of
Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT,
2019. URL https://arxiv.org/abs/1810.04805.
[28] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a Distilled
Version of BERT: Smaller, Faster, Cheaper and Lighter. arXiv Preprint arXiv:1910.01108,
2019.
[29] Colin Raffel, Noam Shazeer, Adam Roberts, et al. Exploring the Limits of Transfer Learning
with a Unified Text-to-Text Transformer. Journal of Machine Learning Research, 2020. URL
https://arxiv.org/abs/1910.10683.
[30] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V.
Le. XLNet: Generalized Autoregressive Pretraining for Language Understanding. In Advances
in Neural Information Processing Systems (NeurIPS), 2019.
579


<!-- page 580 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[31] Alec
Radford,
Jeffrey
Wu,
Rewon
Child,
David
Luen,
Dario
Amodei,
and
Ilya
Sutskever.
Language Models Are Unsupervised Multitask Learners.
OpenAI Blog,
2019.
URL https://cdn.openai.com/better-language-models/language_models_are_
unsupervised_multitask_learners.pdf.
[32] Qwen Team. Qwen2.5: A Party of Foundation Models. arXiv Preprint arXiv:2412.15115, 2024.
URL https://arxiv.org/abs/2412.15115.
[33] Mike Lewis, Yinhan Liu, Naman Goyal, et al. BART: Denoising Sequence-to-Sequence Pre-
Training for Natural Language Generation, Translation, and Comprehension. In Proceedings of
the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020. URL
https://arxiv.org/abs/1910.13461.
[34] Hyung Won Chung, Le Hou, Shayne Longpre, et al. Scaling Instruction-Finetuned Language
Models. Journal of Machine Learning Research, 2024. URL https://arxiv.org/abs/2210.
11416.
[35] Yinhan Liu, Myle Ott, Naman Goyal, et al.
RoBERTa: A Robustly Optimized BERT
Pretraining Approach. arXiv Preprint arXiv:1907.11692, 2019. URL https://arxiv.org/
abs/1907.11692.
[36] John Rupert Firth. A Synopsis of Linguistic Theory, 1930–1955. Studies in Linguistic Analysis,
1957.
[37] Kawin Ethayarajh. How Contextual Are Contextualized Word Representations? Comparing
the Geometry of BERT, ELMo, and GPT-2 Embeddings. In Proceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing (EMNLP), 2019. URL https://arxiv.
org/abs/1909.00512.
[38] Jianlin Su, Jiarun Cao, Weijie Liu, and Yangyiwen Ou. Whitening Sentence Representations
for Better Semantics and Faster Retrieval. arXiv Preprint arXiv:2103.15316, 2021. URL
https://arxiv.org/abs/2103.15316.
[39] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The Long-Document Trans-
former. arXiv Preprint arXiv:2004.05150, 2020. URL https://arxiv.org/abs/2004.05150.
[40] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, et al. Big Bird: Transformers for
Longer Sequences. In Advances in Neural Information Processing Systems (NeurIPS), 2020.
URL https://arxiv.org/abs/2007.14062.
[41] Mandy Guo, Joshua Ainslie, David Uthus, et al. LongT5: Efficient Text-to-Text Transformer
for Long Sequences. Findings of the Association for Computational Linguistics: NAACL 2022,
2022. URL https://arxiv.org/abs/2112.07916.
[42] Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
arXiv Preprint arXiv:2312.00752, 2023. URL https://arxiv.org/abs/2312.00752.
[43] Bo Peng, Eric Alcaide, Quentin Anthony, et al. RWKV: Reinventing RNNs for the Transformer
Era. Findings of the Association for Computational Linguistics: EMNLP 2023, 2023. URL
https://arxiv.org/abs/2305.13048.
[44] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, et al.
H2O: Heavy-Hitter Oracle for Efficient
Generative Inference of Large Language Models. In Advances in Neural Information Processing
Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2306.14048.
[45] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient Streaming
Language Models with Attention Sinks. In International Conference on Learning Representa-
tions (ICLR), 2024. URL https://arxiv.org/abs/2309.17453.
580


<!-- page 581 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[46] Zirui Liu, Jiayi Yuan, Hongye Jin, et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization
for KV Cache.
In International Conference on Machine Learning (ICML), 2024.
URL
https://arxiv.org/abs/2402.02750.
[47] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring Attention with Blockwise Transformers for
Near-Infinite Context. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
URL https://arxiv.org/abs/2310.01889.
[48] BigScience Workshop. BLOOM: A 176B-Parameter Open-Access Multilingual Language Model.
arXiv Preprint arXiv:2211.05100, 2023. URL https://arxiv.org/abs/2211.05100.
[49] MosaicML. MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs. MosaicML
Blog, 2023. URL https://www.mosaicml.com/blog/mpt-7b.
[50] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer:
Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 2024.
[51] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shao. YaRN: Efficient Context Window
Extension of Large Language Models. arXiv Preprint arXiv:2309.00071, 2023.
[52] Ofir Press, Noah A. Smith, and Mike Lewis. Train Short, Test Long: Attention with Linear
Biases Enables Input Length Generalization. ICLR, 2022.
[53] Anthropic. The Claude 3 Model Family: Opus, Sonnet, Haiku. Anthropic Technical Report,
2024. URL https://www.anthropic.com/news/claude-3-family.
[54] Google Gemini Team. Gemini 1.5: Unlocking Multimodal Understanding Across Millions of
Tokens of Context. arXiv Preprint arXiv:2403.05530, 2024. URL https://arxiv.org/abs/
2403.05530.
[55] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending Context Win-
dow of Large Language Models via Positional Interpolation. arXiv Preprint arXiv:2306.15595,
2023. URL https://arxiv.org/abs/2306.15595.
[56] Nelson F. Liu, Kevin Lin, John Hewitt, et al. Lost in the Middle: How Language Models Use
Long Contexts. Transactions of the Association for Computational Linguistics, 2024. URL
https://arxiv.org/abs/2307.03172.
[57] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer Feed-Forward Layers
Are Key-Value Memories. In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing (EMNLP), 2021.
[58] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer Normalization. arXiv Preprint
arXiv:1607.06450, 2016. URL https://arxiv.org/abs/1607.06450.
[59] Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. In Advances in Neural
Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1910.07467.
[60] Meta AI. The Llama 4 Herd: The Beginning of a New Era of Natively Multimodal AI. Meta
AI Blog, 2025. URL https://ai.meta.com/blog/llama-4-multimodal-intelligence/.
[61] Mistral AI.
Mistral Large 2.
Mistral AI Blog, 2024.
URL https://mistral.ai/news/
mistral-large-2407/.
[62] DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv Preprint arXiv:2412.19437, 2024. URL
https://arxiv.org/abs/2412.19437.
[63] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient Streaming
Language Models with Attention Sinks. In Proceedings of the 12th International Conference
on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2309.17453.
581


<!-- page 582 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[64] Yao Fu, Rameswar Panda, Xinyao Niu, et al. Data Engineering for Scaling Language Models
to 128K Context. arXiv Preprint arXiv:2402.10171, 2024. URL https://arxiv.org/abs/
2402.10171.
[65] Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. URL
https://arxiv.org/abs/2312.00752.
[66] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing Multi-
Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. In
Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL),
2019. URL https://arxiv.org/abs/1905.09418.
[67] Catherine Olsson, Nelson Elhage, Neel Nanda, et al. In-Context Learning and Induction
Heads.
Transformer Circuits Thread, 2022.
URL https://transformer-circuits.pub/
2022/in-context-learning-and-induction-heads/index.html.
[68] Zhengbao Wu, Aman Arora, Zhiqiang Wang, Byung-Gon Kim, and Tian Huang. Retrieval
Head Mechanistically Explains Long-Context Factuality. arXiv Preprint arXiv:2404.15574,
2024. URL https://arxiv.org/abs/2404.15574.
[69] Jesse Vig. A Multiscale Visualization of Attention in the Transformer Model. In Proceedings
of the 57th ACL: System Demonstrations, 2019. URL https://arxiv.org/abs/1906.05714.
[70] Samira Abnar and Willem Zuidema. Quantifying Attention Flow in Transformers. In Proceedings
of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
URL https://arxiv.org/abs/2005.00928.
[71] Oren Barkan, Edan Hauon, Avi Caciularu, Ido Dagan, and Noam Koenigstein. Grad-SAM:
Explaining Transformers via Gradient Self-Attention Maps. In Proceedings of the 30th ACM
International Conference on Information and Knowledge Management (CIKM), 2021. URL
https://arxiv.org/abs/2104.13299.
[72] Sarthak Jain and Byron C. Wallace. Attention Is Not Explanation. In Proceedings of the 2019
Conference of the North American Chapter of the Association for Computational Linguistics
(NAACL), 2019. URL https://arxiv.org/abs/1902.10186.
[73] Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse
Autoencoders Find Highly Interpretable Features in Language Models. In Proceedings of
the 12th International Conference on Learning Representations (ICLR), 2024. URL https:
//arxiv.org/abs/2309.08600.
[74] Trenton Bricken, Adly Templeton, Joshua Batson, et al. Towards Monosemanticity: Decom-
posing Language Models with Dictionary Learning. Transformer Circuits Thread, 2023. URL
https://transformer-circuits.pub/2023/monosemantic-features/index.html.
[75] Adly Templeton, Tom Conerly, Jonathan Marcus, et al. Scaling Monosemanticity: Extracting
Interpretable Features from Claude 3 Sonnet.
Transformer Circuits Thread, 2024.
URL
https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html.
[76] Anthropic. Natural Language Autoencoders: Interpreting Neural Networks with Natural
Language Descriptions. Anthropic Research Blog, 2026. URL https://www.anthropic.com/
research/natural-language-autoencoders.
[77] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning Representations
by Back-Propagating Errors. Nature, 1986. URL https://doi.org/10.1038/323533a0.
[78] Herbert Robbins and Sutton Monro. A Stochastic Approximation Method. The Annals of
Mathematical Statistics, 1951.
582


<!-- page 583 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[79] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization.
In
International Conference on Learning Representations (ICLR), 2015. URL https://arxiv.
org/abs/1412.6980.
[80] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. arXiv Preprint
arXiv:1711.05101, 2019. URL https://arxiv.org/abs/1711.05101.
[81] Shengding Hu, Yuge Tu, Xu Han, et al. MiniCPM: Unveiling the Potential of Small Language
Models with Scalable Training Strategies. arXiv Preprint arXiv:2404.06395, 2024. URL
https://arxiv.org/abs/2404.06395.
[82] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.
org/abs/2307.08691.
[83] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao.
FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision. arXiv
Preprint arXiv:2407.08691, 2024. URL https://arxiv.org/abs/2407.08691.
[84] Ted Zadouri, Jay Shah, Ganesh Bikshandi, and Tri Dao. FlashAttention-4: Hardware-Efficient
Attention on Blackwell GPUs with Minimal Software Design. arXiv Preprint arXiv:2603.05451,
2026. URL https://arxiv.org/abs/2603.05451.
[85] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. Training Compute-Optimal
Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS),
2022. URL https://arxiv.org/abs/2203.15556.
[86] Katherine Lee, Daphne Ippolito, Andrew Nystrom, et al. Deduplicating Training Data Makes
Language Models Better. In Proceedings of the 60th Annual Meeting of the ACL, 2022. URL
https://arxiv.org/abs/2107.06499.
[87] Chunting Zhou, Pengfei Liu, Puxin Xu, et al. LIMA: Less Is More for Alignment. In Advances
in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/
2305.11206.
[88] Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, et al. Liger-Kernel: Efficient Triton Kernels for LLM
Training. arXiv Preprint arXiv:2410.10989, 2024. URL https://arxiv.org/abs/2410.10989.
[89] Daniel Han and Michael Han.
Unsloth: Efficient LLM Fine-Tuning, 2024.
URL https:
//github.com/unslothai/unsloth.
[90] PyTorch Team.
Torchtune: PyTorch Native Post-Training Library, 2024.
URL https:
//github.com/pytorch/torchtune.
[91] Neel Jain, Ping yeh Chiang, Yuxin Wen, et al.
NEFTune: Noisy Embeddings Improve
Instruction Finetuning. In International Conference on Learning Representations (ICLR), 2024.
URL https://arxiv.org/abs/2310.05914.
[92] Edward J. Hu, Yelong Shen, Phillip Wallis, et al. LoRA: Low-Rank Adaptation of Large
Language Models. In International Conference on Learning Representations (ICLR), 2022.
URL https://arxiv.org/abs/2106.09685.
[93] Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic Dimensionality Explains
the Effectiveness of Language Model Fine-Tuning. In Proceedings of the 59th Annual Meeting
of the Association for Computational Linguistics (ACL), 2021. URL https://arxiv.org/abs/
2012.13255.
[94] Damjan Kalajdzievski. A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. arXiv
Preprint arXiv:2312.03732, 2023. URL https://arxiv.org/abs/2312.03732.
583


<!-- page 584 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[95] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient
Finetuning of Quantized Language Models. In Advances in Neural Information Processing
Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2305.14314.
[96] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, et al. DoRA: Weight-Decomposed Low-Rank
Adaptation. arXiv Preprint arXiv:2402.09353, 2024. URL https://arxiv.org/abs/2402.
09353.
[97] Soufiane Hayou, Nikhil Ghosh, and Bin Yu. LoRA+: Efficient Low Rank Adaptation of Large
Models. arXiv Preprint arXiv:2402.12354, 2024. URL https://arxiv.org/abs/2402.12354.
[98] Qingru Zhang, Minshuo Chen, Alexander Bukharin, et al.
AdaLoRA: Adaptive Budget
Allocation for Parameter-Efficient Fine-Tuning. arXiv Preprint arXiv:2303.10512, 2023. URL
https://arxiv.org/abs/2303.10512.
[99] Dawid Jan Kopiczko, Tijmen Blankevoort, and Markus Nagel. VeRA: Vector-Based Random
Matrix Adaptation. arXiv Preprint arXiv:2310.11454, 2024. URL https://arxiv.org/abs/
2310.11454.
[100] Neil Houlsby, Andrei Giber, Stanislaw Jastrzebski, et al. Parameter-Efficient Transfer Learning
for NLP. In International Conference on Machine Learning (ICML), 2019. URL https:
//arxiv.org/abs/1902.00751.
[101] Xiang Lisa Li and Percy Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
In Proceedings of the 59th Annual Meeting of the ACL, 2021. URL https://arxiv.org/abs/
2101.00190.
[102] Brian Lester, Rami Al-Rfou, and Noah Constant. The Power of Scale for Parameter-Efficient
Prompt Tuning.
In Proceedings of the 2021 Conference on EMNLP, 2021.
URL https:
//arxiv.org/abs/2104.08691.
[103] Haokun Liu, Derek Tam, Mohammed Muqeeth, et al. Few-Shot Parameter-Efficient Fine-
Tuning Is Better and Cheaper Than in-Context Learning. In Advances in Neural Information
Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2205.05638.
[104] Elad Ben Zaken, Shauli Ravfogel, and Yoav Goldberg. BitFit: Simple Parameter-Efficient
Fine-Tuning for Transformer-Based Masked Language-Models. In Proceedings of the 60th
Annual Meeting of the ACL, 2022. URL https://arxiv.org/abs/2106.10199.
[105] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, et al.
Outrageously Large Neural
Networks: The Sparsely-Gated Mixture-of-Experts Layer. In International Conference on
Learning Representations (ICLR), 2017. URL https://arxiv.org/abs/1701.06538.
[106] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, et al. Mixtral of Experts. arXiv
Preprint arXiv:2401.04088, 2024. URL https://arxiv.org/abs/2401.04088.
[107] Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with Gumbel-Softmax.
In International Conference on Learning Representations (ICLR), 2017.
[108] DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language
Model. arXiv Preprint arXiv:2405.04434, 2024. URL https://arxiv.org/abs/2405.04434.
[109] William Fedus, Barret Zoph, and Noam Shazeer. Switch Transformers: Scaling to Trillion
Parameter Models with Simple and Efficient Sparsity. Journal of Machine Learning Research,
2022.
[110] Databricks.
DBRX: A New State-of-the-Art Open LLM.
Databricks Blog, 2024.
URL
https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm.
584


<!-- page 585 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[111] Jiayi Zhang, Simon Yu, Derek Chong, et al. Verbalized Sampling: How to Mitigate Mode
Collapse and Unlock LLM Diversity. arXiv Preprint arXiv:2510.01171, 2025. URL https:
//arxiv.org/abs/2510.01171.
[112] Ashwin K. Vijayakumar, Michael Cogswell, Ramprasaath R. Selvaraju, et al. Diverse Beam
Search: Decoding Diverse Solutions from Neural Sequence Models. AAAI, 2018.
[113] Minh Nguyen. Min-p Sampling: A Simple Baseline for Better LLM Decoding. arXiv Preprint
arXiv:2310.06022, 2024.
[114] Xian Li, Ari Holtzman, Daniel Fried, et al. Contrastive Decoding: Open-Ended Text Generation
as Optimization. ACL, 2023.
[115] Brandon T. Willard and Rémi Louf. Efficient Guided Generation for Large Language Models.
arXiv Preprint arXiv:2307.09702, 2023. URL https://arxiv.org/abs/2307.09702.
[116] Yixin Dong, Charlie Moon, Yuekai Wang, et al. XGrammar: Flexible and Efficient Structured
Generation Engine for Large Language Models. arXiv Preprint arXiv:2411.15100, 2024.
[117] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An Explanation of
in-Context Learning as Implicit Bayesian Inference. In Proceedings of the 10th International
Conference on Learning Representations (ICLR), 2022. URL https://arxiv.org/abs/2111.
02080.
[118] Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau.
Function Vectors in Large Language Models. In Proceedings of the 12th International Conference
on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2310.15213.
[119] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically
Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity.
In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics
(ACL), 2022. URL https://arxiv.org/abs/2104.08786.
[120] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen.
What Makes Good in-Context Examples for GPT-3? In Proceedings of Deep Learning Inside
Out (DeeLIO), ACL Workshop, 2022. URL https://arxiv.org/abs/2101.06804.
[121] Sewon Min, Xinxi Lyu, Ari Holtzman, et al. Rethinking the Role of Demonstrations: What
Makes in-Context Learning Work? In Proceedings of the 2022 Conference on Empirical Methods
in Natural Language Processing (EMNLP), 2022. URL https://arxiv.org/abs/2202.12837.
[122] Jason Wei, Xuezhi Wang, Dale Schuurmans, et al.
Chain-of-Thought Prompting Elicits
Reasoning in Large Language Models. In Advances in Neural Information Processing Systems
(NeurIPS), 2022. URL https://arxiv.org/abs/2201.11903.
[123] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
Language Models Are Zero-Shot Reasoners. In Advances in Neural Information Processing
Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2205.11916.
[124] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, et al. Self-Consistency Improves Chain
of Thought Reasoning in Language Models. In Proceedings of the 11th International Conference
on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2203.11171.
[125] Shunyu Yao, Dian Yu, Jeffrey Zhao, et al. Tree of Thoughts: Deliberate Problem Solving with
Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS),
2023. URL https://arxiv.org/abs/2305.10601.
[126] Lei Wang, Wanyu Xu, Yiber Lan, et al. Plan-and-Solve Prompting: Improving Zero-Shot
Chain-of-Thought Reasoning by Large Language Models. In Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (ACL), 2023. URL https://arxiv.
org/abs/2305.04091.
585


<!-- page 586 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[127] Shunyu Yao, Jeffrey Zhao, Dian Yu, et al. ReAct: Synergizing Reasoning and Acting in
Language Models. In International Conference on Learning Representations (ICLR), 2023.
URL https://arxiv.org/abs/2210.03629.
[128] Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems
(NeurIPS), 2020. URL https://arxiv.org/abs/2005.11401.
[129] Yuntao Bai, Andy Jones, Kamal Ndousse, et al. Constitutional AI: Harmlessness from AI
Feedback.
arXiv Preprint arXiv:2212.08073, 2022.
URL https://arxiv.org/abs/2212.
08073.
[130] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, et al. Large Language Models Are Human-
Level Prompt Engineers. In Proceedings of the 11th International Conference on Learning
Representations (ICLR), 2023. URL https://arxiv.org/abs/2211.01910.
[131] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, et al. DSPy: Compiling Declarative
Language Model Calls into Self-Improving Pipelines. In Proceedings of the 12th International
Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2310.
03714.
[132] Chengrun Yang, Xuezhi Wang, Yifeng Lu, et al. Large Language Models as Optimizers. In
Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024.
URL https://arxiv.org/abs/2309.03409.
[133] Jingwen Yang, Yuxuan Zhu, Binyuan Wang, et al. ARQ: Attentive Reasoning Queries for
Multi-Hop Question Answering over Long Contexts. arXiv Preprint arXiv:2501.08290, 2025.
URL https://arxiv.org/abs/2501.08290.
[134] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh.
GPTQ: Accurate
Post-Training Quantization for Generative Pre-Trained Transformers.
arXiv Preprint
arXiv:2210.17323, 2023. URL https://arxiv.org/abs/2210.17323.
[135] Ji Lin, Jiaming Tang, Haotian Tang, et al. AWQ: Activation-Aware Weight Quantization
for LLM Compression and Acceleration. In Proceedings of Machine Learning and Systems
(MLSys), 2024. URL https://arxiv.org/abs/2306.00978.
[136] Georgi Gerganov. GGUF: GPT-Generated Unified Format (llama.cpp), 2023. URL https:
//github.com/ggerganov/llama.cpp.
[137] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han.
SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.
In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. URL
https://arxiv.org/abs/2211.10438.
[138] Zechun Liu, Barlas Oguz, Changsheng Zhao, et al.
LLM-QAT: Data-Free Quantization
Aware Training for Large Language Models. arXiv Preprint arXiv:2305.17888, 2023. URL
https://arxiv.org/abs/2305.17888.
[139] Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babber, and Dan
Alistarh. Extreme Compression of Large Language Models via Additive Quantization. arXiv
Preprint arXiv:2401.06118, 2024. URL https://arxiv.org/abs/2401.06118.
[140] Elias Frantar and Dan Alistarh. SparseGPT: Massive Language Models Can Be Accurately
Pruned in One-Shot. In Proceedings of the 40th International Conference on Machine Learning
(ICML), 2023. URL https://arxiv.org/abs/2301.00774.
[141] Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. A Simple and Effective Pruning
Approach for Large Language Models. In International Conference on Learning Representations
(ICLR), 2024. URL https://arxiv.org/abs/2306.11695.
586


<!-- page 587 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[142] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network.
arXiv Preprint arXiv:1503.02531, 2015.
[143] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast Inference from Transformers via
Speculative Decoding. In Proceedings of the 40th International Conference on Machine Learning
(ICML), 2023. URL https://arxiv.org/abs/2211.17192.
[144] Tianle Cai, Yuhong Li, Zhengyang Geng, et al. Medusa: Simple LLM Inference Acceleration
Framework with Multiple Decoding Heads. In Proceedings of the 41st International Conference
on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2401.10774.
[145] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE: Speculative Sampling
Requires Rethinking Feature Uncertainty. In Proceedings of the 41st International Conference
on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2401.15077.
[146] Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the Sequential Dependency of
LLM Inference Using Lookahead Decoding. arXiv Preprint arXiv:2402.02057, 2024. URL
https://arxiv.org/abs/2402.02057.
[147] Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve.
Better & Faster Large Language Models via Multi-Token Prediction. In Proceedings of the 41st
International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/
2404.19737.
[148] Ziwei Ji, Nayeon Lee, Rita Frieske, et al.
Survey of Hallucination in Natural Language
Generation. ACM Computing Surveys, 2023. URL https://arxiv.org/abs/2202.03629.
[149] Saurav Kadavath, Tom Conerly, Amanda Askell, et al. Language Models (Mostly) Know What
They Know. arXiv Preprint arXiv:2207.05221, 2022. URL https://arxiv.org/abs/2207.
05221.
[150] Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. SelfCheckGPT: Zero-Resource Black-
Box Hallucination Detection for Generative Large Language Models. In Proceedings of EMNLP,
2023. URL https://arxiv.org/abs/2303.08896.
[151] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic Uncertainty: Linguistic Invariances
for Uncertainty Estimation in Natural Language Generation. In International Conference on
Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2302.09664.
[152] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He.
DoLA: Decoding by Contrasting Layers Improves Factuality in Large Language Models. In
Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024.
URL https://arxiv.org/abs/2309.03883.
[153] Isabel O. Gallegos, Ryan A. Rossi, Joe Barber, Shanshan Tong, et al. Bias and Fairness in
Large Language Models: A Survey. Computational Linguistics, 2024. URL https://arxiv.
org/abs/2309.00770.
[154] Nicholas Carlini, Florian Tramer, Eric Wallace, et al. Extracting Training Data from Large
Language Models. USENIX Security Symposium, 2021. URL https://arxiv.org/abs/2012.
07805.
[155] Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. Universal and Transferable
Adversarial Attacks on Aligned Language Models. arXiv Preprint arXiv:2307.15043, 2023.
URL https://arxiv.org/abs/2307.15043.
[156] Ethan Perez, Saffron Huang, Francis Song, et al. Red Teaming Language Models with Language
Models. In Proceedings of EMNLP, 2022. URL https://arxiv.org/abs/2202.03286.
587


<!-- page 588 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[157] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. Efficient Memory Management for Large
Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th
Symposium on Operating Systems Principles (SOSP), 2023. URL https://arxiv.org/abs/
2309.06180.
[158] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction, 2018.
URL http://incompleteideas.net/book/the-book-2nd.html.
[159] Richard S Sutton. Learning to Predict by the Methods of Temporal Differences. Machine
Learning, 1988.
[160] Christopher J. C. H Watkins. Learning from Delayed Rewards. 1989.
[161] Gavin A. Rummery and Mahesan Niranjan. On-Line q-Learning Using Connectionist Systems,
1994.
[162] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al. Human-Level Control Through
Deep Reinforcement Learning. Nature, 2015. URL https://www.nature.com/articles/
nature14236.
[163] Long-Ji Lin. Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and
Teaching. Machine Learning, 1992.
[164] Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized Experience Replay.
In Proceedings of the 4th International Conference on Learning Representations (ICLR), 2016.
URL https://arxiv.org/abs/1511.05952.
[165] Ronald J Williams. Simple Statistical Gradient-Following Algorithms for Connectionist Rein-
forcement Learning. Machine Learning, 1992. URL https://link.springer.com/article/
10.1007/BF00992696.
[166] Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, et al. Asynchronous Methods for
Deep Reinforcement Learning. In Proceedings of the 33rd International Conference on Machine
Learning (ICML), 2016. URL https://arxiv.org/abs/1602.01783.
[167] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust
Region Policy Optimization. In Proceedings of the 32nd International Conference on Machine
Learning (ICML), 2015. URL https://arxiv.org/abs/1502.05477.
[168] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
Policy Optimization Algorithms. arXiv Preprint arXiv:1707.06347, 2017. URL https://arxiv.
org/abs/1707.06347.
[169] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-
Dimensional Continuous Control Using Generalized Advantage Estimation. In Proceedings
of the 4th International Conference on Learning Representations (ICLR), 2016. URL https:
//arxiv.org/abs/1506.02438.
[170] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft Actor-Critic: Off-Policy
Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. In Proceedings of
the 35th International Conference on Machine Learning (ICML), 2018. URL https://arxiv.
org/abs/1801.01290.
[171] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, et al. Mastering Atari, Go, Chess
and Shogi by Planning with a Learned Model. Nature, 2020. URL https://arxiv.org/abs/
1911.08265.
[172] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to Control:
Learning Behaviors by Latent Imagination. In Proceedings of the 8th International Conference
on Learning Representations (ICLR), 2020. URL https://arxiv.org/abs/1912.01603.
588


<!-- page 589 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[173] Andrew Y. Ng, Daishi Harada, and Stuart J. Russell.
Policy Invariance Under Reward
Transformations: Theory and Application to Reward Shaping. In Proceedings of the 16th
International Conference on Machine Learning (ICML), 1999.
[174] Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, et al. Fine-Tuning Language Models from
Human Preferences. arXiv Preprint arXiv:1909.08593, 2019. URL https://arxiv.org/abs/
1909.08593.
[175] Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei.
Deep Reinforcement Learning from Human Preferences. In Advances in Neural Information
Processing Systems (NeurIPS), 2017. URL https://arxiv.org/abs/1706.03741.
[176] Leandro von Werra, Younes Belkada, Lewis Tunstall, et al. TRL: Transformer Reinforcement
Learning, 2022. URL https://github.com/huggingface/trl.
[177] Junkang Wang, Jianfei Duan, Yue Liu, Zhangchen Yue, Hanghang Tong, and James Wang.
Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence
Constraints. In Proceedings of the 12th International Conference on Learning Representations
(ICLR), 2024. URL https://arxiv.org/abs/2309.16240.
[178] Sayak Ray Chowdhury, Anush Chakraborty, Srinadh Natarajan, Alekh Agarwal, and David
Sontag. Provably Robust DPO: Aligning Language Models with Noisy Feedback. arXiv Preprint
arXiv:2403.00409, 2024. URL https://arxiv.org/abs/2403.00409.
[179] Alexander Gorbatenko. Online DPO with Synchronised Reference Model Updates. TRL
Documentation, 2024.
[180] Jiatao Ji, Adam Fisch, Jason Weston, and Sainbayar Sukhbaatar. Towards Exact Optimization
of Language Model Alignment. arXiv Preprint arXiv:2402.05369, 2024. URL https://arxiv.
org/abs/2402.05369.
[181] Huayu Chen, Guande Zheng, Yimeng Kim, and Yiwen Chen. Noise Contrastive Alignment
of Language Models with Explicit Rewards. arXiv Preprint arXiv:2402.05369, 2024. URL
https://arxiv.org/abs/2402.05369.
[182] Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J.
Liu.
SLiC-HF: Sequence Likelihood Calibration with Human Feedback.
arXiv Preprint
arXiv:2305.10425, 2023. URL https://arxiv.org/abs/2305.10425.
[183] Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple Preference Optimization with a
Reference-Free Reward. In Advances in Neural Information Processing Systems (NeurIPS),
2024. URL https://arxiv.org/abs/2405.14734.
[184] Qiying Yu, Zheng Sun, Shang Wen, et al. DAPO: An Open-Source LLM Reinforcement Learning
System. arXiv Preprint arXiv:2503.14476, 2025. URL https://arxiv.org/abs/2503.14476.
[185] Zhiyu Chen, Yiwei Deng, Ruiqi Zhang, Hao Sun, and Weizhu Chen. GSPO: Sequence-Level
Policy Optimization for Language Model Alignment. arXiv Preprint arXiv:2502.12459, 2025.
URL https://arxiv.org/abs/2502.12459.
[186] Yihao Liu, Lefan Han, Yifan Tan, et al.
Understanding and Mitigating the Pretraining
Distribution Bias in GRPO. arXiv Preprint arXiv:2505.07888, 2025. URL https://arxiv.
org/abs/2505.07888.
[187] Haoxiang Xu, Hongyuan Zhao, Ying Liu, and Dong Wei. It Takes Two: Pairwise Preference
Optimization with Two Rollouts.
arXiv Preprint arXiv:2505.07856, 2025.
URL https:
//arxiv.org/abs/2505.07856.
589


<!-- page 590 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[188] Chi Han, Minlie Li, and Wenting Chen. SAPO: Soft Adaptive Policy Optimization for Efficient
LLM Alignment. arXiv Preprint arXiv:2503.01739, 2025. URL https://arxiv.org/abs/
2503.01739.
[189] Yanqi Zhong, Yifu Chen, Zijun Li, and Minlie Chen. Importance Sampling Corrections for Large
Language Model Alignment with Asynchronous Generation. arXiv Preprint arXiv:2503.09057,
2025. URL https://arxiv.org/abs/2503.09057.
[190] Zhixun Luo, Jiaqi Shi, Tao Yu, and Minlie Chen. VESPO: Variational Sequence-Level Soft
Policy Optimization for LLM Alignment.
arXiv Preprint arXiv:2505.07508, 2025.
URL
https://arxiv.org/abs/2505.07508.
[191] Yanxu An, Li Shen, Yifan Xu, and Xinmei Liu. DPPO: Direct Divergence-Based Policy
Optimization for Language Model Alignment. arXiv Preprint arXiv:2503.14532, 2025. URL
https://arxiv.org/abs/2503.14532.
[192] Zhenyu Luo, Ziyan Chen, Yulun Jiang, et al. ScaleRL: Scaling Reinforcement Learning for
LLM Reasoning. arXiv Preprint arXiv:2505.16356, 2025. URL https://arxiv.org/abs/2505.
16356.
[193] Yanqi Zhong, Jiaqi Shi, Yifu Chen, and Minlie Chen. GDPO: Learning to Directly Align
Language Models with Group-Decoupled Reward. arXiv Preprint arXiv:2501.17888, 2025.
URL https://arxiv.org/abs/2501.17888.
[194] Seokhyun Choi, Hyunji Park, Doyoung Moon, Kyuyoung Kim, and Edward Choi. GOPO:
Group Ordinal Policy Optimization for LLM Alignment with Non-Verifiable Rewards. arXiv
Preprint arXiv:2505.12948, 2025. URL https://arxiv.org/abs/2505.12948.
[195] Shangmin Guo, Biao Zhang, Tianlin Liu, et al. Direct Language Model Alignment from Online
AI Feedback. arXiv Preprint arXiv:2402.04792, 2024. URL https://arxiv.org/abs/2402.
04792.
[196] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, et al. WebGPT: Browser-Assisted Question-
Answering with Human Feedback.
arXiv Preprint arXiv:2112.09332, 2021.
URL https:
//arxiv.org/abs/2112.09332.
[197] Leo Gao, John Schulman, and Jacob Hilton. Scaling Laws for Reward Model Overoptimization.
In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
[198] Ralph Allan Bradley and Milton E. Terry. Rank Analysis of Incomplete Block Designs: I. The
Method of Paired Comparisons. Biometrika, 1952. URL https://www.jstor.org/stable/
2334029.
[199] R. L Plackett. The Analysis of Permutations. Journal of the Royal Statistical Society: Series
C (Applied Statistics), 1975.
[200] Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. Listwise Approach to
Learning to Rank: Theory and Algorithm. In Proceedings of the 25th International Conference
on Machine Learning (ICML), 2008.
[201] Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. Learning to Rank: From
Pairwise Approach to Listwise Approach. In Proceedings of the 24th International Conference
on Machine Learning (ICML), 2007.
[202] Christopher J. C. Burges, Robert Ragno, and Quoc V. Le. Learning to Rank with Nonsmooth
Cost Functions. In Advances in Neural Information Processing Systems (NeurIPS), 2006.
[203] Chris Burges, Tal Shaked, Erin Renshaw, et al. Learning to Rank Using Gradient Descent. In
Proceedings of the 22nd International Conference on Machine Learning (ICML), 2005.
590


<!-- page 591 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[204] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, et al. Overcoming Catastrophic Forget-
ting in Neural Networks. In Proceedings of the National Academy of Sciences, 2017.
[205] Shen Li, Yanli Zhao, Rohan Varma, et al. PyTorch Distributed: Experiences on Accelerating
Data Parallel Training. In Proceedings of the VLDB Endowment, 2020.
[206] Alexander Sergeev and Mike Del Balso. Horovod: Fast and Easy Distributed Deep Learning in
TensorFlow. arXiv Preprint arXiv:1802.05799, 2018.
[207] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan
Catanzaro. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
Parallelism. arXiv Preprint arXiv:1909.08053, 2019.
[208] Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, et al. Reducing Activation Recomputa-
tion in Large Transformer Models. In Proceedings of Machine Learning and Systems (MLSys),
2023.
[209] Yanping Huang, Youlong Cheng, Ankur Bapna, et al. GPipe: Efficient Training of Giant
Neural Networks Using Pipeline Parallelism. In Advances in Neural Information Processing
Systems (NeurIPS), 2019.
[210] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, et al. PipeDream: Generalized Pipeline
Parallelism for DNN Training. In Proceedings of the 27th ACM Symposium on Operating
Systems Principles (SOSP), 2019.
[211] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, et al. Efficient Large-Scale Language
Model Training on GPU Clusters Using Megatron-LM. arXiv Preprint arXiv:2104.04473, 2021.
[212] Penghui Qi, Xinyi Wan, Guangxing Huang, and Min Lin. Zero Bubble Pipeline Parallelism.
arXiv Preprint arXiv:2401.10241, 2023.
[213] Samyam Rajbhandari, Jeff Rasley, Olatunji Rber, and Yuxiong He. ZeRO: Memory Opti-
mizations Toward Training Trillion Parameter Models. arXiv Preprint arXiv:1910.02054,
2020.
[214] Yanli Zhao, Andrew Gu, Rohan Varma, et al. PyTorch FSDP: Experiences on Scaling Fully
Sharded Data Parallel. In Proceedings of the VLDB Endowment, 2023.
[215] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca:
A Distributed Serving System for Transformer-Based Generative Models. In Proceedings of the
16th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2022.
[216] Zhewei Yao, Samyam Rajbhandari, Reza Yazdani Aminabadi, et al. DeepSpeed-Chat: Easy,
Fast and Affordable RLHF Training of ChatGPT-Like Models at All Scales. arXiv Preprint
arXiv:2308.01320, 2023.
[217] Jian Hu, Xibin Tao, Weixun Zhu, Sicheng Yang, Jingwen Liu, and Zilin Li. OpenRLHF: An Easy-
to-Use, Scalable and High-Performance RLHF Framework. arXiv Preprint arXiv:2405.11143,
2024.
[218] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training Deep Nets with Sublinear
Memory Cost. arXiv Preprint arXiv:1604.06174, 2016.
[219] Paulius Micikevicius, Sharan Narang, Jonah Alben, et al. Mixed Precision Training. In
International Conference on Learning Representations (ICLR), 2018.
[220] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. ZeRO-
Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning. arXiv Preprint
arXiv:2104.07857, 2021.
591


<!-- page 592 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[221] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. PaLM: Scaling Language Modeling
with Pathways. arXiv Preprint arXiv:2204.02311, 2022.
[222] Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An Empirical Model
of Large-Batch Training. arXiv Preprint arXiv:1812.06162, 2018.
[223] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Bootstrapping Reasoning
with Reasoning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL
https://arxiv.org/abs/2203.14465.
[224] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao.
Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural
Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2303.11366.
[225] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang.
Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models.
arXiv Preprint arXiv:2310.04406, 2024.
[226] Pranav Putta, Edmund Mills, Naman Garg, et al. Agent Q: Advanced Reasoning and Learning
for Autonomous AI Agents. arXiv Preprint arXiv:2408.07199, 2024.
[227] Xingyao Wang, Boxuan Ding, Ziniu Hoang, et al. OpenHands: An Open Platform for AI
Software Developers as Generalist Agents. arXiv Preprint arXiv:2407.16741, 2024.
[228] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, et al. Voyager: An Open-Ended Embodied Agent
with Large Language Models. arXiv Preprint arXiv:2305.16291, 2023. URL https://arxiv.
org/abs/2305.16291.
[229] Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Hoi. CodeRL:
Mastering Code Generation Through Pretrained Models and Deep Reinforcement Learning. In
Advances in Neural Information Processing Systems (NeurIPS), 2023.
[230] Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D. Goodman.
Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking.
arXiv
Preprint arXiv:2403.09629, 2024. URL https://arxiv.org/abs/2403.09629.
[231] Arian Hosseini, Xingdi Yuan, Pascal Poupart, Adam Trischler, and Yoshua Bengio. V-STaR:
Training Verifiers for Self-Taught Reasoners. arXiv Preprint arXiv:2402.06457, 2024.
[232] John Yang, Carlos E. Jimenez, Alexander Wettig, et al. SWE-agent: Agent-Computer Interfaces
Enable Automated Software Engineering. In Advances in Neural Information Processing Systems
(NeurIPS), 2024. URL https://arxiv.org/abs/2405.15793.
[233] Xiao Yu, Baolin Peng, Ruize Xu, et al. Reinforcement World Model Learning for LLM-Based
Agents. arXiv Preprint arXiv:2602.05842, 2026.
[234] Shunyu Yao, Dian Yu, Jeffrey Zhao, et al. Tree of Thoughts: Deliberate Problem Solving with
Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS),
2024. URL https://arxiv.org/abs/2305.10601.
[235] Maciej Besta, Nils Blach, Ales Kubicek, et al. Graph of Thoughts: Solving Elaborate Problems
with Large Language Models. In Proceedings of the AAAI Conference on Artificial Intelligence,
2024.
[236] Nisan Stiennon, Long Ouyang, Jeffrey Wu, et al.
Learning to Summarize from Human
Feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL
https://arxiv.org/abs/2009.01325.
[237] Levente Kocsis and Csaba Szepesvári. Bandit Based Monte-Carlo Planning. In European
Conference on Machine Learning (ECML), 2006.
592


<!-- page 593 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[238] Google DeepMind.
AlphaProof and AlphaGeometry 2:
Solving Olympiad Geometry
Without Human Demonstrations, 2024. URL https://deepmind.google/discover/blog/
ai-solves-imo-problems-at-silver-medal-level/.
[239] Zhenting Qi, Mingyuan Wan, Jialin Cao, and Min Lin. Mutual Reasoning Makes Smaller LLMs
Stronger Problem-Solvers. arXiv Preprint arXiv:2408.06195, 2024.
[240] Aman Madaan, Niket Tandon, Prakhar Gupta, et al. Self-Refine: Iterative Refinement with
Self-Feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
[241] OpenAI.
Learning to Reason with LLMs, 2024.
URL https://openai.com/index/
learning-to-reason-with-llms/.
[242] OpenAI. OpenAI o3 and o4-mini System Card, 2025. URL https://openai.com/index/
o3-and-o4-mini-system-card/.
[243] Hunter Lightman, Vineet Kosaraju, Yura Burda, et al. Let’s Verify Step by Step. arXiv
Preprint arXiv:2305.20050, 2023.
[244] Qwen Team. QwQ: Reflect Deeply on the Boundaries of the Unknown. Qwen Blog, 2024. URL
https://qwenlm.github.io/blog/qwq-32b-preview/.
[245] Qwen Team. Qwen3 Technical Report. arXiv Preprint arXiv:2505.09388, 2025.
[246] Peiyi Wang, Lei Li, Zhihong Shao, et al. Math-Shepherd: Verify and Reinforce LLMs Step-
by-Step Without Human Annotations. arXiv Preprint arXiv:2312.08935, 2024. URL https:
//arxiv.org/abs/2312.08935.
[247] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, et al.
Tülu 3: Pushing Frontiers
in Open Language Model Post-Training.
arXiv Preprint arXiv:2411.15124, 2024.
URL
https://arxiv.org/abs/2411.15124.
[248] Yiwei Qin, Xuefeng Li, Haoyang Zou, et al. O1 Replication Journey: A Strategic Progress
Report. arXiv Preprint arXiv:2410.18982, 2024. URL https://arxiv.org/abs/2410.18982.
[249] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM Test-Time Com-
pute Optimally Can Be More Effective Than Scaling Model Parameters. arXiv Preprint
arXiv:2408.03314, 2024.
[250] Zhenyu Wu, Qinghua Hu, Yin Zhang, Yiming Gao, and Jiangtao Chen. An Empirical Analysis
of Compute-Optimal Inference for Problem-Solving with Language Models. arXiv Preprint
arXiv:2408.00724, 2024.
[251] Jared Kaplan, Sam McCandlish, Tom Henighan, et al. Scaling Laws for Neural Language
Models. arXiv Preprint arXiv:2001.08361, 2020. URL https://arxiv.org/abs/2001.08361.
[252] Jacob Cohen. A Coefficient of Agreement for Nominal Scales. Educational and Psychological
Measurement, 1960.
[253] Joseph L Fleiss. Measuring Nominal Scale Agreement Among Many Raters. Psychological
Bulletin, 1971.
[254] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On Calibration of Modern Neural
Networks. In International Conference on Machine Learning (ICML), 2017.
[255] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, et al. Self-Instruct: Aligning Language
Models with Self-Generated Instructions.
arXiv Preprint arXiv:2212.10560, 2022.
URL
https://arxiv.org/abs/2212.10560.
593


<!-- page 594 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[256] Can Xu, Qingfeng Sun, Kai Zheng, et al. WizardLM: Empowering Large Language Models to
Follow Complex Instructions. arXiv Preprint arXiv:2304.12244, 2023. URL https://arxiv.
org/abs/2304.12244.
[257] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. Judging LLM-as-a-Judge with MT-Bench
and Chatbot Arena. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
URL https://arxiv.org/abs/2306.05685.
[258] Arpad E Elo. The Rating of Chess Players, Past and Present, 1978.
[259] Ralf
Herbrich,
Tom
Minka,
and
Thore
Graepel.
TrueSkill™:
A
Bayesian
Skill
Rating
System.
In
Advances
in
Neural
Information
Processing
Systems
(NeurIPS),
2006.
URL
https://proceedings.neurips.cc/paper/2006/hash/
f44ee263952e65b3610b8ba51229d1f9-Abstract.html.
[260] Edwin B Wilson. Probable Inference, the Law of Succession, and Statistical Inference. Journal
of the American Statistical Association, 1927. URL https://www.jstor.org/stable/2276774.
[261] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu.
BLEU: A Method for
Automatic Evaluation of Machine Translation. In Proceedings of the 40th Annual Meeting
of the Association for Computational Linguistics (ACL), 2002. URL https://aclanthology.
org/P02-1040/.
[262] Chin-Yew Lin.
ROUGE: A Package for Automatic Evaluation of Summaries.
In Text
Summarization Branches Out: Proceedings of the ACL-04 Workshop, 2004.
URL https:
//aclanthology.org/W04-1013/.
[263] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. BERTScore:
Evaluating Text Generation with BERT. In International Conference on Learning Representa-
tions (ICLR), 2020. URL https://arxiv.org/abs/1904.09675.
[264] Satanjeev Banerjee and Alon Lavie. METEOR: An Automatic Metric for MT Evaluation
with Improved Correlation with Human Judgments. In Proceedings of the ACL Workshop on
Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization,
2005. URL https://aclanthology.org/W05-0909/.
[265] Mark Chen, Jerry Tworek, Heewoo Jun, et al. Evaluating Large Language Models Trained on
Code. arXiv Preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.
[266] Carlos E. Jimenez, John Yang, Alexander Wettig, et al. SWE-bench: Can Language Models
Resolve Real-World GitHub Issues? In International Conference on Learning Representations
(ICLR), 2024. URL https://arxiv.org/abs/2310.06770.
[267] Shuyan Zhou, Frank F. Xu, Hao Zhu, et al. WebArena: A Realistic Web Environment for
Building Autonomous Agents. In International Conference on Learning Representations (ICLR),
2024. URL https://arxiv.org/abs/2307.13854.
[268] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and
Matthew Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive
Learning. In International Conference on Learning Representations (ICLR), 2021.
[269] Xiao Liu, Hao Yu, Hanchen Zhang, et al. AgentBench: Evaluating LLMs as Agents. arXiv
Preprint arXiv:2308.03688, 2023.
[270] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-Eval:
NLG Evaluation Using GPT-4 with Better Human Alignment. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. URL
https://arxiv.org/abs/2303.16634.
594


<!-- page 595 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[271] Dan Hendrycks, Collin Burns, Steven Basart, et al. Measuring Massive Multitask Language
Understanding. In International Conference on Learning Representations (ICLR), 2021.
[272] Charles A. E Goodhart. Problems of Monetary Management: The U.K. Experience. Monetary
Theory and Practice, 1984.
[273] Stephen Robertson and Hugo Zaragoza. The Probabilistic Relevance Framework: BM25 and
Beyond. Foundations and Trends in Information Retrieval, 2009.
[274] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, et al. Dense Passage Retrieval for Open-Domain
Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), 2020. URL https://arxiv.org/abs/2004.04906.
[275] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-Scale Similarity Search with GPUs.
IEEE Transactions on Big Data, 2021.
[276] Yu. A. Malkov and D. A. Yashunin. Efficient and Robust Approximate Nearest Neighbor
Search Using Hierarchical Navigable Small World Graphs. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 2020.
[277] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher. Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods. In Proceedings of the 32nd
International ACM SIGIR Conference, 2009.
[278] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. SPLADE: Sparse Lexical
and Expansion Model for First Stage Ranking. In Proceedings of the 44th International ACM
SIGIR Conference on Research and Development in Information Retrieval, 2021.
[279] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. SPLADE
v2:
Sparse Lexical and Expansion Model for Information Retrieval.
arXiv Preprint
arXiv:2109.10086, 2021.
[280] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. Document Ranking with a
Pretrained Sequence-to-Sequence Model. Findings of EMNLP, 2020.
[281] Payal Bajaj, Daniel Campos, Nick Craswell, et al. MS MARCO: A Human Generated MAchine
Reading COmprehension Dataset. arXiv Preprint arXiv:1611.09268, 2016.
[282] Omar Khattab and Matei Zaharia. ColBERT: Efficient and Effective Passage Search via
Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM
SIGIR Conference, 2020. URL https://arxiv.org/abs/2004.12832.
[283] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia.
ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In Proceedings
of the 2022 Conference of the North American Chapter of the Association for Computational
Linguistics (NAACL), 2022.
[284] Karen Sparck Jones. A Statistical Interpretation of Term Specificity and Its Application in
Retrieval. Journal of Documentation, 1972.
[285] Rodrigo Nogueira and Kyunghyun Cho. Passage Re-Ranking with BERT. arXiv Preprint
arXiv:1901.04085, 2019.
[286] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise Zero-Shot Dense Retrieval
Without Relevance Labels. In Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (ACL), 2023. URL https://arxiv.org/abs/2212.10496.
[287] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning
to Retrieve, Generate, and Critique Through Self-Reflection. arXiv Preprint arXiv:2310.11511,
2023. URL https://arxiv.org/abs/2310.11511.
595


<!-- page 596 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[288] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective Retrieval Augmented
Generation. arXiv Preprint arXiv:2401.15884, 2024. URL https://arxiv.org/abs/2401.
15884.
[289] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-RAG:
Learning to Adapt Retrieval-Augmented Large Language Models Through Question Complexity.
arXiv Preprint arXiv:2403.14403, 2024. URL https://arxiv.org/abs/2403.14403.
[290] Darren Edge, Ha Trinh, Newman Cheng, et al.
From Local to Global: A Graph RAG
Approach to Query-Focused Summarization. arXiv Preprint arXiv:2404.16130, 2024. URL
https://arxiv.org/abs/2404.16130.
[291] Adrian H Rackauckas. RAG-Fusion: A New Take on Retrieval-Augmented Generation, 2024.
[292] Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, and Vijai
Mohan. REFRAG: Rethinking RAG Based Decoding. arXiv Preprint arXiv:2509.01092, 2025.
[293] Bowen Jin, Hansi Zeng, et al. Search-R1: Training LLMs to Reason and Leverage Search
Engines with Reinforcement Learning. arXiv Preprint arXiv:2503.09516, 2025.
[294] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, et al. Natural Questions: A Bench-
mark for Question Answering Research. Transactions of the Association for Computational
Linguistics, 2019.
[295] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A Large Scale
Distantly Supervised Challenge Dataset for Reading Comprehension. In Proceedings of ACL,
2017.
[296] Zhilin Yang, Peng Qi, Saizheng Zhang, et al. HotpotQA: A Dataset for Diverse, Explainable
Multi-Hop Question Answering. In Proceedings of EMNLP, 2018.
[297] Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. RAGAs: Automated
Evaluation of Retrieval Augmented Generation. arXiv Preprint arXiv:2309.15217, 2023. URL
https://arxiv.org/abs/2309.15217.
[298] Nelson F. Liu, Kevin Lin, John Hewitt, et al. Lost in the Middle: How Language Models Use
Long Contexts. Transactions of the Association for Computational Linguistics, 2024. URL
https://arxiv.org/abs/2307.03172.
[299] Chankyu Lee, Rajarshi Roy, Menber Xu, et al. NV-Embed: Improved Techniques for Training
LLMs as Generalist Embedding Models. arXiv Preprint arXiv:2405.17428, 2024.
[300] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang.
Towards General Text Embeddings with Multi-Stage Contrastive Learning. arXiv Preprint
arXiv:2308.03281, 2023.
[301] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. BGE M3-
Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through
Self-Knowledge Distillation. arXiv Preprint arXiv:2402.03216, 2024.
[302] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-Pack: Packaged Resources
to Advance General Chinese Embedding. arXiv Preprint arXiv:2309.07597, 2023.
[303] Tianjun Zhang, Shishir G. Patil, Naman Jain, et al. RAFT: Adapting Language Model to
Domain Specific RAG. arXiv Preprint arXiv:2403.10131, 2024. URL https://arxiv.org/
abs/2403.10131.
[304] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM:
Retrieval-Augmented Language Model Pre-Training. In Proceedings of the 37th International
Conference on Machine Learning (ICML), 2020. URL https://arxiv.org/abs/2002.08909.
596


<!-- page 597 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[305] Endel Tulving. Memory and Consciousness. Canadian Psychology / Psychologie Canadienne,
1985.
[306] Larry R Squire. Declarative and Nondeclarative Memory: Multiple Brain Systems Supporting
Learning and Memory. Journal of Cognitive Neuroscience, 1992. URL https://doi.org/10.
1162/jocn.1992.4.3.232.
[307] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, et al. Show Your Work: Scratchpads
for Intermediate Computation with Language Models. arXiv Preprint arXiv:2112.00114, 2021.
URL https://arxiv.org/abs/2112.00114.
[308] Ruiqi Guo, Philip Sun, Erik Lindgren, et al. Accelerating Large-Scale Inference with Anisotropic
Vector Quantization. In International Conference on Machine Learning (ICML), 2020.
[309] Jianlv Chen, Shitao Luo, Mingzheng Zhang, Zheng Liu, Yingxia Xiao, and Defu Han. Hybrid
Retrieval for Open-Domain Question Answering. arXiv Preprint arXiv:2210.06029, 2022. URL
https://arxiv.org/abs/2210.06029.
[310] Tiago Forte. Building a Second Brain: A Proven Method to Organize Your Digital Life and
Unlock Your Creative Potential, 2022.
[311] Steve Harris and Andy Seaborne. SPARQL 1.1 Query Language, 2013. URL https://www.w3.
org/TR/sparql11-query/.
[312] Nadime Francis, Alastair Green, Paolo Guagliardo, et al. Cypher: An Evolving Query Language
for Property Graphs. In Proceedings of the 2018 International Conference on Management of
Data (SIGMOD), 2018.
[313] Timothée Lacroix, Guillaume Obozinski, and Nicolas Usunier. Tensor Decompositions for Tem-
poral Knowledge Base Completion. In International Conference on Learning Representations
(ICLR), 2020. URL https://arxiv.org/abs/2004.04926.
[314] Jason Weston, Sumit Chopra, and Antoine Bordes. Memory Networks. In International
Conference on Learning Representations (ICLR), 2015. URL https://arxiv.org/abs/1410.
3916.
[315] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-End Memory
Networks. In Advances in Neural Information Processing Systems (NeurIPS), 2015. URL
https://arxiv.org/abs/1503.08895.
[316] Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, and Joseph E.
Gonzalez. MemGPT: Towards LLMs as Operating Systems. arXiv Preprint arXiv:2310.08560,
2023. URL https://arxiv.org/abs/2310.08560.
[317] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-Mem:
Agentic Memory for LLM Agents. In Advances in Neural Information Processing Systems
(NeurIPS), 2025.
[318] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0:
Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv Preprint
arXiv:2504.19413, 2025. URL https://arxiv.org/abs/2504.19413.
[319] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang,
and Michael S. Bernstein. Generative Agents: Interactive Simulacra of Human Behavior. In
Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology
(UIST), 2023. URL https://arxiv.org/abs/2304.03442.
[320] Hermann Ebbinghaus. Über Das Gedächtnis: Untersuchungen Zur Experimentellen Psychologie,
1885.
597


<!-- page 598 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[321] Barbara Hayes-Roth. A Blackboard Architecture for Control. Artificial Intelligence, 1985. URL
https://doi.org/10.1016/0004-3702(85)90063-3.
[322] Marcin Andrychowicz, Filip Wolski, Alex Ray, et al. Hindsight Experience Replay. In Advances
in Neural Information Processing Systems (NeurIPS), 2017.
[323] Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel.
RL2:
Fast Reinforcement Learning via Slow Reinforcement Learning.
arXiv Preprint
arXiv:1611.02779, 2016.
[324] Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell.
Curiosity-Driven
Exploration by Self-Supervised Prediction. In Proceedings of the 34th International Conference
on Machine Learning (ICML), 2017.
[325] Alex Graves, Greg Wayne, Malcolm Reynolds, et al. Hybrid Computing Using a Neural Network
with Dynamic External Memory. Nature, 2016.
[326] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval:
Benchmarking Chat Assistants on Long-Term Interactive Memory. In International Conference
on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2410.10813.
[327] Adyasha Maharana, Dong-Ho Lee, Sergey Tuber, Mohit Ruber, Francesco Barbieri, and Mohit
Bansal. LoCoMo: Long-Context Conversation with Memory Operations. In Proceedings of the
2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.
[328] Xinrong Zhang, Yingfa Chen, Shengding Hu, et al. InfiniteBench: Extending Long Context
Evaluation Beyond 100K Tokens. In Proceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics (ACL), 2024. URL https://arxiv.org/abs/2402.13718.
[329] Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Griffiths. Cognitive
Architectures for Language Agents. Transactions on Machine Learning Research (TMLR),
2024. URL https://arxiv.org/abs/2309.02427.
[330] Kevin Lin, Charlie Snell, Yu Wang, et al. Sleep-Time Compute: Beyond Inference Scaling
at Test-Time. arXiv Preprint arXiv:2504.13171, 2025. URL https://arxiv.org/abs/2504.
13171.
[331] Alex L. Zhang, Seyyed Hasan Mahdavi, Percy Liang, and Tatsunori Hashimoto. Recursive
Language Models. arXiv Preprint arXiv:2512.24601, 2025. URL https://arxiv.org/abs/
2512.24601.
[332] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, et al. Toolformer: Language Models Can Teach
Themselves to Use Tools. In Advances in Neural Information Processing Systems, 2023.
[333] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large Language
Model Connected with Massive APIs. In Advances in Neural Information Processing Systems
(NeurIPS), 2024. URL https://arxiv.org/abs/2305.15334.
[334] Yujia Qin, Shihao Liang, Yining Ye, et al. ToolLLM: Facilitating Large Language Models
to Master 16000+ Real-World APIs. In Proceedings of the 12th International Conference on
Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2307.16789.
[335] Anthropic. Model Context Protocol, 2024. URL https://modelcontextprotocol.io.
[336] OpenAI. Swarm: An Educational Framework for Lightweight Multi-Agent Orchestration, 2024.
URL https://github.com/openai/swarm.
[337] LangChain Inc. LangGraph: Build Stateful Multi-Actor Applications with LLMs, 2024. URL
https://github.com/langchain-ai/langgraph.
598


<!-- page 599 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[338] Qingyun Wu, Gagan Bansal, Jieyu Zhang, et al. AutoGen: Enabling Next-Gen LLM Applica-
tions via Multi-Agent Conversation. arXiv Preprint arXiv:2308.08155, 2023.
[339] Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to Use Large Language
Models While Reducing Cost and Improving Performance. arXiv Preprint arXiv:2305.05176,
2023. URL https://arxiv.org/abs/2305.05176.
[340] Harrison Chase. LangChain, 2022. URL https://github.com/langchain-ai/langchain.
[341] João Moura. CrewAI: Framework for Orchestrating Role-Playing Autonomous AI Agents, 2023.
URL https://github.com/crewAIInc/crewAI.
[342] Anthropic. Building Effective Agents, 2024. URL https://www.anthropic.com/research/
building-effective-agents.
[343] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, et al. Self-Consistency Improves
Chain of Thought Reasoning in Language Models. In International Conference on Learning
Representations (ICLR), 2023. URL https://arxiv.org/abs/2203.11171.
[344] Greg Brockman, Vicki Cheung, Ludwig Pettersson, et al. OpenAI Gym, 2016.
[345] Jaeseok Yoo and Dongmin Shin.
Adaptive Episode Length Adjustment for Multi-Agent
Reinforcement Learning. In Proceedings of the 24th International Conference on Autonomous
Agents and Multiagent Systems (AAMAS), 2025.
[346] Zhuokai Liu, Hao Dong, et al. DLER: Doing Length Penalty Right—Incentivizing More
Intelligence Per Token. arXiv Preprint arXiv:2510.15110, 2025. URL https://arxiv.org/
abs/2510.15110.
[347] Qiguang Liu et al. Answer Convergence as a Signal for Early Stopping in Reasoning. In
Proceedings of EMNLP, 2025. URL https://aclanthology.org/2025.emnlp-main.904/.
[348] Jiangfei Mei et al. APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-Tail
Generation. arXiv Preprint arXiv:2509.18521, 2025. URL https://arxiv.org/abs/2509.
18521.
[349] Qinghao Hu, Shang Yang, et al. Taming the Long-Tail: Efficient Reasoning RL Training with
Adaptive Drafter. arXiv Preprint arXiv:2511.16665, 2025. URL https://arxiv.org/abs/
2511.16665.
[350] Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel.
Prioritized Level Replay.
In
Proceedings of the 38th International Conference on Machine Learning (ICML), 2021. URL
https://arxiv.org/abs/2010.03934.
[351] Michael Dennis, Natasha Jaques, Eugene Vinitsky, et al. Emergent Complexity and Zero-Shot
Transfer via Unsupervised Environment Design. In Advances in Neural Information Processing
Systems (NeurIPS), 2020.
[352] Yifan Wang et al. Improving Data Efficiency for LLM Reinforcement Fine-Tuning via Difficulty-
Targeted Online Data Selection.
In Advances in Neural Information Processing Systems
(NeurIPS), 2025. URL https://arxiv.org/abs/2506.05316.
[353] Xingyu Liu et al. Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive
Difficulty Curriculum Learning. arXiv Preprint arXiv:2505.08364, 2025. URL https://arxiv.
org/abs/2505.08364.
[354] Jing Yu Koh, Robert Lo, Lawrence Jang, et al. VisualWebArena: Evaluating Multimodal Agents
on Realistic Visual Web Tasks. In Proceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics (ACL), 2024. URL https://arxiv.org/abs/2401.13649.
599


<!-- page 600 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[355] Xiang Deng, Yu Gu, Boyuan Zheng, et al. Mind2Web: Towards a Generalist Agent for
the Web. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL
https://arxiv.org/abs/2306.06070.
[356] Tianbao Xie, Danyang Zhang, Jixuan Chen, et al. OSWorld: Benchmarking Multimodal Agents
for Open-Ended Tasks in Real Computer Environments. In Advances in Neural Information
Processing Systems, 2024.
[357] Rogerio Bonatti, Dan Zhao, Francesco Bonacci, et al. Windows Agent Arena: Evaluating
Multi-Modal OS Agents at Scale.
arXiv Preprint arXiv:2409.08264, 2024.
URL https:
//arxiv.org/abs/2409.08264.
[358] Jakub Lála, Odhran O’Donoghue, Aleksandar Shtedritski, Sam Cox, Samuel G. Rodriques, and
Andrew D. White. PaperQA: Retrieval-Augmented Generative Agent for Scientific Research.
arXiv Preprint arXiv:2312.07559, 2023. URL https://arxiv.org/abs/2312.07559.
[359] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The
AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv Preprint
arXiv:2408.06292, 2024. URL https://arxiv.org/abs/2408.06292.
[360] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. MLAgentBench: Evaluating Language
Agents on Machine Learning Experimentation.
In Proceedings of the 41st International
Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2310.03302.
[361] Heinrich Küttler, Nantas Nardelli, Alexander H. Miller, et al. The NetHack Learning En-
vironment. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL
https://arxiv.org/abs/2006.13760.
[362] Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas
Scialom. GAIA: A Benchmark for General AI Assistants. In International Conference on
Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2311.12983.
[363] Mike Lewis, Denis Yarats, Yann N. Dauphin, Devi Parikh, and Dhruv Batra. Deal or No Deal?
End-to-End Learning for Negotiation Dialogues. In Proceedings of EMNLP, 2017.
[364] Kushal Chawla, Jaysa Ramirez, Rene Clever, Gale Lucas, Jonathan May, and Jonathan Gratch.
CaSiNo: A Corpus of Campsite Negotiation Dialogues for Automatic Negotiation Systems. In
Proceedings of NAACL, 2021.
[365] Sirui Hong, Mingchen Zhuge, Jonathan Chen, et al. MetaGPT: Meta Programming for a
Multi-Agent Collaborative Framework, 2024.
[366] Hugging Face. OpenEnv: An Interface Library for RL Post-Training with Environments, 2025.
URL https://github.com/huggingface/OpenEnv.
[367] Mark Towers, Ariel Kwiatkowski, Jordan Terry, et al. Gymnasium: A Standard Interface
for Reinforcement Learning Environments. NeurIPS Datasets and Benchmarks, 2024. URL
https://arxiv.org/abs/2407.17032.
[368] Zhiheng Xi, Yiwen Ding, Wenxiang Chen, et al. AgentGym: Evolving Large Language Model-
Based Agents Across Diverse Environments. arXiv Preprint arXiv:2406.04151, 2024. URL
https://arxiv.org/abs/2406.04151.
[369] Thibault Le Sellier De Chezelles, Maxime Gasse, Alexandre Drouin, Massimo Caccia, et al.
The BrowserGym Ecosystem for Web Agent Research. arXiv Preprint arXiv:2412.05467, 2024.
URL https://arxiv.org/abs/2412.05467.
[370] Meta PyTorch Team.
TorchForge: PyTorch-Native Post-Training at Scale, 2025.
URL
https://github.com/meta-pytorch/torchforge.
600


<!-- page 601 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[371] JSON-RPC Working Group. JSON-RPC 2.0 Specification, 2010. URL https://www.jsonrpc.
org/specification.
[372] Google.
Agent2Agent (A2A) Protocol, 2025.
URL https://developers.google.com/
agent2agent.
[373] Tom Preston-Werner. Semantic Versioning 2.0.0, 2024. URL https://semver.org/.
[374] Reid G Smith. The Contract Net Protocol: High-Level Communication and Control in a
Distributed Problem Solver. IEEE Transactions on Computers, 1980. URL https://doi.org/
10.1109/TC.1980.1675516.
[375] Michael Jones, John Bradley, and Nat Sakimura. JSON Web Token (JWT), 2015. URL
https://datatracker.ietf.org/doc/html/rfc7519.
[376] Brian Campbell, John Bradley, Nat Sakimura, and Torsten Lodderstedt. OAuth 2.0 Mutual-TLS
Client Authentication and Certificate-Bound Access Tokens, 2020. URL https://datatracker.
ietf.org/doc/html/rfc8705.
[377] World Wide Web Consortium. Decentralized Identifiers (DIDs) v1.0, 2022. URL https:
//www.w3.org/TR/did-core/.
[378] Eric Rescorla. The Transport Layer Security (TLS) Protocol Version 1.3, 2018. URL https:
//datatracker.ietf.org/doc/html/rfc8446.
[379] Dick Hardt. The OAuth 2.0 Authorization Framework, 2012. URL https://datatracker.
ietf.org/doc/html/rfc6749.
[380] Gerhard Weiss. Multiagent Systems: A Modern Approach to Distributed Artificial Intelligence,
1999.
[381] Michael Wooldridge. An Introduction to MultiAgent Systems, 2009.
[382] Edmund H. Durfee, Victor R. Lesser, and Daniel D. Corkill. Trends in Cooperative Distributed
Problem Solving.
IEEE Transactions on Knowledge and Data Engineering, 1989.
URL
https://doi.org/10.1109/69.43404.
[383] Edward de Bono. Six Thinking Hats, 1985.
[384] Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving
Factuality and Reasoning in Language Models Through Multiagent Debate. In Proceedings of
the 41st International Conference on Machine Learning (ICML), 2023. URL https://arxiv.
org/abs/2305.14325.
[385] Lloyd S Shapley. Stochastic Games. In Proceedings of the National Academy of Sciences, 1953.
URL https://www.pnas.org/doi/10.1073/pnas.39.10.1095.
[386] Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch.
Multi-
Agent Actor-Critic for Mixed Cooperative-Competitive Environments. In Advances in Neural
Information Processing Systems (NeurIPS), 2017. URL https://arxiv.org/abs/1706.02275.
[387] Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob
Foerster, and Shimon Whiteson. QMIX: Monotonic Value Function Factorisation for Deep
Multi-Agent Reinforcement Learning. In Proceedings of the 35th International Conference on
Machine Learning (ICML), 2018. URL https://arxiv.org/abs/1803.11605.
[388] Sainbayar Sukhbaatar, Arthur Szlam, and Rob Fergus. Learning Multiagent Communication
with Backpropagation. In Advances in Neural Information Processing Systems (NeurIPS),
2016. URL https://arxiv.org/abs/1605.07736.
601


<!-- page 602 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[389] Abhishek Das, Théophile Gerber, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra.
TarMAC: Targeted Multi-Agent Communication. In Proceedings of the 36th International
Conference on Machine Learning (ICML), 2019. URL https://arxiv.org/abs/1810.11187.
[390] Angeliki Lazaridou and Marco Baroni. Emergent Multi-Agent Communication in the Deep
Learning Era. arXiv Preprint arXiv:2006.02419, 2020. URL https://arxiv.org/abs/2006.
02419.
[391] Max Jaderberg, Wojciech M. Czarnecki, Iain Dunning, et al. Human-Level Performance in
3D Multiplayer Games with Population-Based Reinforcement Learning. Science, 2019. URL
https://www.science.org/doi/10.1126/science.aau6249.
[392] Yoav Shoham and Kevin Leyton-Brown. Multiagent Systems: Algorithmic, Game-Theoretic,
and Logical Foundations, 2008. URL http://www.masfoundations.org/.
[393] Kaiqing Zhang, Zhuoran Yang, and Tamer Başar. Multi-Agent Reinforcement Learning: A
Selective Overview of Theories and Algorithms. Handbook of Reinforcement Learning and
Control, 2021. URL https://arxiv.org/abs/1911.10635.
[394] Noam Nisan, Tim Roughgarden, Éva Tardos, and Vijay V. Vazirani. Algorithmic Game Theory,
2007. URL https://www.cs.cmu.edu/~sandholm/cs15-892F13/algorithmic-game-theory.
pdf.
[395] OpenAI.
OpenAI
Agents
SDK,
2025.
URL
https://github.com/openai/
openai-agents-python.
[396] Microsoft. Semantic Kernel: SDK for Integrating AI Models into Applications, 2023. URL
https://github.com/microsoft/semantic-kernel.
[397] Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Prompting Is Programming: A Query
Language for Large Language Models. In Proceedings of PLDI, 2023.
[398] Raja Parasuraman and Victor Riley. Humans and Automation: Use, Misuse, Disuse, Abuse.
Human Factors, 1997. URL https://doi.org/10.1518/001872097778543886.
[399] Vercel. Vercel AI SDK, 2024. URL https://sdk.vercel.ai.
[400] Chainlit. Chainlit: Build Production-Ready Conversational AI Applications, 2024. URL
https://chainlit.io.
[401] Abubakar Abid, Ali Abdalla, Ali Abid, Dawood Khan, Abdulrahman Alfozan, and James
Zou.
Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild, 2019.
URL
https://arxiv.org/abs/1906.02569.
[402] Streamlit Inc.
Streamlit: The Fastest Way to Build and Share Data Apps, 2024.
URL
https://streamlit.io.
[403] LangChain Inc. LangGraph Studio: The First Agent IDE, 2024. URL https://github.com/
langchain-ai/langgraph-studio.
[404] Javier García and Fernando Fernández. A Comprehensive Survey on Safe Reinforcement
Learning. Journal of Machine Learning Research, 2015.
[405] Geoffrey Irving, Paul Christiano, and Dario Amodei. AI Safety via Debate. arXiv Preprint
arXiv:1805.00899, 2018. URL https://arxiv.org/abs/1805.00899.
[406] Evan Hubinger, Carson Denison, Jesse Mu, et al. Sleeper Agents: Training Deceptive LLMs
That Persist Through Safety Training. arXiv Preprint arXiv:2401.05566, 2024.
602


<!-- page 603 -->
H. Roitman
—
The Hitchhiker’s Guide to Agentic AI: From Foundations to Systems
[407] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan
Mané.
Concrete Problems in AI Safety.
arXiv Preprint arXiv:1606.06565, 2016.
URL
https://arxiv.org/abs/1606.06565.
[408] Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and
Mario Fritz. Not What You’ve Signed up for: Compromising Real-World LLM-Integrated
Applications with Indirect Prompt Injection. In Proceedings of the 16th ACM Workshop on
Artificial Intelligence and Security (AISec), 2023.
[409] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger.
Defining and
Characterizing Reward Hacking. In Advances in Neural Information Processing Systems, 2022.
URL https://arxiv.org/abs/2209.13085.
[410] Seungone Kim, Se June Jang, et al. Distilling Step-by-Step! Outperforming Larger Language
Models with Less Training Data and Smaller Model Sizes. Findings of the ACL, 2023. URL
https://arxiv.org/abs/2305.02301.
603
