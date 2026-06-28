# 第 29 章 结语与未来方向

## 29.1 小结

本指南完整梳理了从 Transformer 基础,到用于对齐的强化学习(RL),再到自主智能体系统构建的完整脉络。贯穿所有章节浮现出的核心主题如下:

1. **对齐是一个系统问题**。仅有一个好的损失函数是远远不够的。生产级 RLHF 需要同时管理 4 个以上模型、在数百张 GPU 上分布式调度计算、处理容错,并监控奖励黑客(reward hacking)——这一切都要同时进行。
2. **不存在唯一最佳的方法**。PPO 仍是追求最高质量时的黄金标准,但需要巨大的工程投入。DPO 及其变体为基础设施有限的团队提供了极具吸引力的折中方案。GRPO 在可验证奖励(verifiable-reward)领域弥合了二者之间的空白。正确的选择取决于你的数据、算力预算和质量门槛。
3. **推理源自奖励**。DeepSeek-R1 证明,思维链(Chain-of-Thought)、自我验证(self-verification)和回溯(backtracking)可以从简单的二元奖励信号与组相对优化(group-relative optimization)中涌现出来——无需对推理过程的显式示范。测试时计算扩展(test-time compute scaling)意味着,具备更多“思考”的更小模型可以匹敌更大的模型。
4. **标准解锁生态**。MCP 将工具集成问题从 $N \times M$ 降维为 $N + M$。A2A 让由不同团队构建的智能体无需共享内部实现即可协作。这些协议之于智能体 AI(Agentic AI),恰如 HTTP 之于万维网——它们是开放生态的使能基础设施。
5. **智能体是自然的下一步**。一旦模型完成对齐,前沿就从“单次响应有多好?”转向“模型能否自主求解多步问题?”这要求新的训练范式(带有环境奖励的智能体式 RL)、新的基础设施(运行框架、工具协议、记忆系统),以及新的评估方法(轨迹级基准)。
6. **评估驱动一切**。没有严格的评估——从奖励模型验证到智能体任务成功率,从污染检测(contamination detection)到 LLM-as-Judge(LLM 作为评判者)校准——进步将不可度量,退化也将无从察觉。你选择的基准,塑造了你构建的系统。
7. **简单才能扩展**。最可靠的生产级智能体,使用的是满足需求的最简架构——在自主循环之前先用提示链(prompt chaining)与路由(routing),在多智能体蜂群之前先用单智能体。复杂性应当由被验证的需求来赚取,而非一开始就堆砌。

## 29.2 前路:开放挑战

### 29.2.1 从交互中学习

当前的 RLHF 流水线 [9] 将对齐视为一次性的训练阶段。未来的方向指向从部署中持续学习:智能体从每一次用户交互、工具失败和环境观察中改进自身——既不发生灾难性遗忘(catastrophic forgetting)[204],也不出现奖励漂移(reward drift)。关键开放问题包括:

- 在非平稳(non-stationary)奖励分布下的在线学习。
- 生产环境中的安全探索(safe exploration)[404](在学习过程中避免有害动作)。
- 在长智能体轨迹(数百次工具调用)上的高效信用分配(credit assignment)。

### 29.2.2 可扩展监督

随着智能体能力增强,人类监督成为瓶颈。当前方法(RLHF [9]、Constitutional AI [129])依赖人类评估模型输出——但当模型输出超出人类理解力时,会发生什么?

- **递归奖励建模(recursive reward modeling)** [175]:用 AI 辅助人类评估 AI。
- **辩论与放大(debate and amplification)** [405]:两个模型相互辩论;由人类判定哪一方的论证更有说服力。
- **基于过程的监督(process-based supervision)** [243]:奖励正确的推理步骤,而不仅仅是最终答案。
- **机制可解释性(mechanistic interpretability)** [67]:理解模型内部在做什么,而不仅仅看它输出了什么。

### 29.2.3 世界模型与规划

当前智能体是反应式的(reactive)——它们一次一步地观察并响应。未来的智能体将需要内部世界模型 [172] 以支持前瞻式规划(lookahead planning):

- 在执行动作之前预测其后果。
- 对可能的动作序列进行树搜索(类似 AlphaGo [19] 和 MuZero [171],但面向开放式任务)。
- 从交互轨迹中学习环境动力学(environment dynamics)。

### 29.2.4 多智能体生态

A2A 协议 [372] 与多智能体框架预示着一个未来:成百上千个专业化智能体相互协作、协商、委派——形成一个“智能体经济”(economy of agents)[394]。开放挑战包括:

- 在拥有不同委托人(principals)的智能体之间建立信任与验证。
- 竞争场景中的涌现合作(emergent cooperation)与涌现欺骗(emergent deception)[406]。
- 用于资源分配(算力、工具访问、优先级)的市场机制。
- 治理:当一条由 10 个智能体构成的链产生有害后果时,谁来负责?[407]

### 29.2.5 智能体安全与信任

自主智能体继承了其底层 LLM 的每一项安全漏洞——再加上由工具访问、多智能体委派和持久化记忆所带来的新攻击面(第 19–21 章)。关键未解问题包括:

- **大规模提示注入(prompt injection at scale)** [408]:随着智能体消费不可信内容(网页、邮件、API 响应),间接提示注入(indirect prompt injection)演变为系统性风险。目前尚不存在稳健的防御手段。
- **混淆代理(confused deputy)攻击**:拥有合法凭证的智能体可能被诱骗,代攻击者(藏身于数据流之中)误用这些凭证 [335]。
- **不致残的沙箱化**:最小特权执行(least-privilege execution)约束了智能体能做什么,但过于严苛的沙箱会抵消智能体价值。如何找到恰当的边界,是一个开放的系统设计问题。
- **审计与归因(attribution)**:当智能体链跨越多个组织(经由 A2A [372])时,追溯“谁授权了哪个动作”在架构层面仍未解决。
- **信任校准(trust calibration)**:智能体必须学会何时不去信任——工具响应是否真实,另一智能体的声称是否经过验证。

### 29.2.6 超越基准的评估

第 14 章表明,基准塑造了我们构建的系统——然而当前的评估存在关键缺口:

- **真实世界部署指标**:像 SWE-bench [266] 和 GAIA [362] 这样的基准测量的是孤立任务;生产级智能体面对的是模糊的目标、不断变化的需求,以及多轮恢复(multi-turn recovery)。
- **奖励模型有效性**:RLHF 假设奖励模型能够捕捉人类偏好,但奖励黑客 [409] 与分布漂移(distributional shift)在大规模下会动摇这一假设。
- **成本-质量前沿**:两个智能体可能达到相同准确率,但其中一个的 token 成本是另一个的 10 倍。评估必须变得成本感知(cost-aware)。
- **分布漂移下的安全性**:在测试中安全的智能体,可能在全新输入上表现不安全。智能体规模下的对抗性评估(adversarial evaluation)[156] 与红队(red-teaming)仍不成熟。

### 29.2.7 效率与可及性

用 RLHF 训练一个 70B 模型的成本在 1 万至 10 万美元之间。运行自主智能体完成一个复杂任务的成本在 1 至 50 美元之间。要让智能体 AI 产生广泛影响,需要:

- 将智能体能力从大模型蒸馏(distillation)到小模型 [142, 410]。
- 更高效的 RL 算法(更少样本、更低方差)[168]。
- 无需云端往返(round-trips)即可运行的端侧智能体(on-device agents)。
- 在智能体任务上匹敌专有模型质量的开源权重模型(open-weight models)[15]。

## 29.3 扩展阅读

### 29.3.1 奠基性论文

- **Attention Is All You Need** [6] —— Transformer 架构。
- **RLHF / InstructGPT** [9] —— 首次大规模 RLHF 部署。
- **PPO** [168] —— 近端策略优化(Proximal Policy Optimization)。
- **DPO** [10] —— 直接偏好优化(Direct Preference Optimization)。
- **GRPO / DeepSeek-R1** [14, 15] —— 组相对策略优化(Group Relative Policy Optimization)与涌现推理。
- **ReAct** [127] —— 面向 LLM 智能体的推理 + 行动(Reasoning + Acting)框架。
- **Toolformer** [332] —— 教会 LLM 使用工具。
- **RAG** [128] —— 检索增强生成(Retrieval-Augmented Generation)。

### 29.3.2 系统与扩展

- **Megatron-LM** [207] —— 张量并行与流水线并行。
- **DeepSpeed ZeRO** [213] —— 内存高效的分布式训练。
- **vLLM** [157] —— 面向高效 LLM 服务(serving)的 PagedAttention。
- **Flash Attention** [7] —— 感知 IO 的精确注意力(exact attention)。

### 29.3.3 智能体 AI

- **Building Effective Agents** [342] —— 设计模式与原则。
- **Voyager** [228] —— Minecraft 中带有技能库的开放式智能体。
- **SWE-bench** [266] —— 自主软件工程基准。
- **OSWorld** [356] —— 完整计算机使用(computer-use)基准。
- **GAIA** [362] —— 面向真实世界任务的通用 AI 助手基准。
- **MemGPT** [316] —— 操作系统启发的、面向无界上下文(unbounded context)的记忆管理。
- **Model Context Protocol** [335] —— 工具集成的开放标准。
- **Agent-to-Agent Protocol** [372] —— 智能体间通信标准。

### 29.3.4 对齐与安全

- **Constitutional AI** [129] —— 自监督对齐(self-supervised alignment)。
- **Sleeper Agents** [406] —— 欺骗性对齐(deceptive alignment)的隐忧。
- **Reflexion** [224] —— 从言语化自我反思(verbal self-reflection)中学习。
- **Indirect Prompt Injection** [408] —— LLM 集成应用的安全风险。

### 29.3.5 在线资源

- **HuggingFace TRL**:https://github.com/huggingface/trl —— 生产级 RL 库。
- **LangGraph**:https://github.com/langchain-ai/langgraph —— 智能体工作流图。
- **OpenAI Agents SDK**:https://github.com/openai/openai-agents-python —— 官方智能体框架。
- **DeepSpeed-Chat**:https://github.com/microsoft/DeepSpeedExamples —— 端到端 RLHF 流水线。
- **DSPy**:https://github.com/stanfordnlp/dspy —— 声明式提示优化。
- **AutoGen**:https://github.com/microsoft/autogen —— 多智能体会话框架。

> "预测未来的最佳方式,就是亲手去创造它。"
> —— Alan Kay

## 参考文献

[1] Michael Wooldridge, Nicholas R. Jennings, and David Kinny. The Gaia Methodology for Agent-Oriented Analysis and Design. Autonomous Agents and Multi-Agent Systems, 2000.

[2] Fabio Luigi Bellifemine, Giovanni Caire, and Dominic Greenwood. JADE: Developing Multi-Agent Systems with JADE, 2007.

[3] Foundation for Intelligent Physical Agents. FIPA ACL Message Structure Specification, 2002. URL http://www.fipa.org/specs/fipa00061/.

[4] Tim Berners-Lee, James Hendler, and Ora Lassila. The Semantic Web. Scientific American, 2001.

[5] Avigdor Gal, Ateret Anaby-Tavor, Alberto Trombetta, and Danilo Montesi. A Framework for Modeling and Evaluating Automatic Semantic Reconciliation. In Proceedings of the 31st International Conference on Very Large Data Bases (VLDB), 2005. URL https://link.springer.com/chapter/10.1007/11896548_42.

[6] Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. Attention Is All You Need. In Advances in Neural Information Processing Systems (NeurIPS), 2017. URL https://arxiv.org/abs/1706.03762.

[7] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2205.14135.

[8] Edward J. Hu, Yelong Shen, Phillip Wallis, et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv Preprint arXiv:2106.09685, 2022.

[9] Long Ouyang, Jeffrey Wu, Xu Jiang, et al. Training Language Models to Follow Instructions with Human Feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2203.02155.

[10] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2305.18290.

[11] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO: Model Alignment as Prospect Theoretic Optimization. arXiv Preprint arXiv:2402.01306, 2024. URL https://arxiv.org/abs/2402.01306.

[12] Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, et al. A General Theoretical Paradigm to Understand Learning from Human Feedback. arXiv Preprint arXiv:2310.12036, 2024. URL https://arxiv.org/abs/2310.12036.

[13] Jiwoo Hong, Noah Lee, and James Thorne. ORPO: Monolithic Preference Optimization Without Reference Model. arXiv Preprint arXiv:2403.07691, 2024. URL https://arxiv.org/abs/2403.07691.

[14] Zhihong Shao, Peiyi Wang, Qihao Zhu, et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv Preprint arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.

[15] DeepSeek-AI, Daya Guo, Dejian Yang, et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv Preprint arXiv:2501.12948, 2025. URL https://arxiv.org/abs/2501.12948.

[16] Murray Campbell, A. Joseph Hoane Jr., and Feng hsiung Hsu. Deep Blue. Artificial Intelligence, 2002.

[17] David Ferrucci, Eric Brown, Jennifer Chu-Carroll, et al. Building Watson: An Overview of the DeepQA Project. AI Magazine, 2010.

[18] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS, 2012.

[19] David Silver, Aja Huang, Chris J. Maddison, et al. Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature, 2016. URL https://www.nature.com/articles/nature16961.

[20] David Silver, Julian Schrittwieser, Karen Simonyan, et al. Mastering the Game of Go Without Human Knowledge. Nature, 2017. URL https://www.nature.com/articles/nature24270.

[21] Tom Brown, Benjamin Mann, Nick Ryder, et al. Language Models Are Few-Shot Learners. NeurIPS, 2020.

[22] John Jumper, Richard Evans, Alexander Pritzel, et al. Highly Accurate Protein Structure Prediction with AlphaFold. Nature, 2021.

[23] OpenAI. GPT-4 Technical Report. arXiv Preprint arXiv:2303.08774, 2023.

[24] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural Machine Translation of Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of the ACL, 2016. URL https://arxiv.org/abs/1508.07909.

[25] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The Llama 3 Herd of Models. arXiv Preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.

[26] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. Mistral 7B. arXiv Preprint arXiv:2310.06825, 2023. URL https://arxiv.org/abs/2310.06825.

[27] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT, 2019. URL https://arxiv.org/abs/1810.04805.

[28] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. arXiv Preprint arXiv:1910.01108, 2019.

[29] Colin Raffel, Noam Shazeer, Adam Roberts, et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research, 2020. URL https://arxiv.org/abs/1910.10683.

[30] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. XLNet: Generalized Autoregressive Pretraining for Language Understanding. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

[31] Alec Radford, Jeffrey Wu, Rewon Child, David Luen, Dario Amodei, and Ilya Sutskever. Language Models Are Unsupervised Multitask Learners. OpenAI Blog, 2019. URL https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

[32] Qwen Team. Qwen2.5: A Party of Foundation Models. arXiv Preprint arXiv:2412.15115, 2024. URL https://arxiv.org/abs/2412.15115.

[33] Mike Lewis, Yinhan Liu, Naman Goyal, et al. BART: Denoising Sequence-to-Sequence Pre-Training for Natural Language Generation, Translation, and Comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020. URL https://arxiv.org/abs/1910.13461.

[34] Hyung Won Chung, Le Hou, Shayne Longpre, et al. Scaling Instruction-Finetuned Language Models. Journal of Machine Learning Research, 2024. URL https://arxiv.org/abs/2210.11416.

[35] Yinhan Liu, Myle Ott, Naman Goyal, et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv Preprint arXiv:1907.11692, 2019. URL https://arxiv.org/abs/1907.11692.

[36] John Rupert Firth. A Synopsis of Linguistic Theory, 1930–1955. Studies in Linguistic Analysis, 1957.

[37] Kawin Ethayarajh. How Contextual Are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019. URL https://arxiv.org/abs/1909.00512.

[38] Jianlin Su, Jiarun Cao, Weijie Liu, and Yangyiwen Ou. Whitening Sentence Representations for Better Semantics and Faster Retrieval. arXiv Preprint arXiv:2103.15316, 2021. URL https://arxiv.org/abs/2103.15316.

[39] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The Long-Document Transformer. arXiv Preprint arXiv:2004.05150, 2020. URL https://arxiv.org/abs/2004.05150.

[40] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, et al. Big Bird: Transformers for Longer Sequences. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv.org/abs/2007.14062.

[41] Mandy Guo, Joshua Ainslie, David Uthus, et al. LongT5: Efficient Text-to-Text Transformer for Long Sequences. Findings of the Association for Computational Linguistics: NAACL 2022, 2022. URL https://arxiv.org/abs/2112.07916.

[42] Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv Preprint arXiv:2312.00752, 2023. URL https://arxiv.org/abs/2312.00752.

[43] Bo Peng, Eric Alcaide, Quentin Anthony, et al. RWKV: Reinventing RNNs for the Transformer Era. Findings of the Association for Computational Linguistics: EMNLP 2023, 2023. URL https://arxiv.org/abs/2305.13048.

[44] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2306.14048.

[45] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient Streaming Language Models with Attention Sinks. In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2309.17453.

[46] Zirui Liu, Jiayi Yuan, Hongye Jin, et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. In International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2402.02750.

[47] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring Attention with Blockwise Transformers for Near-Infinite Context. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2310.01889.

[48] BigScience Workshop. BLOOM: A 176B-Parameter Open-Access Multilingual Language Model. arXiv Preprint arXiv:2211.05100, 2023. URL https://arxiv.org/abs/2211.05100.

[49] MosaicML. MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs. MosaicML Blog, 2023. URL https://www.mosaicml.com/blog/mpt-7b.

[50] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 2024.

[51] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shao. YaRN: Efficient Context Window Extension of Large Language Models. arXiv Preprint arXiv:2309.00071, 2023.

[52] Ofir Press, Noah A. Smith, and Mike Lewis. Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization. ICLR, 2022.

[53] Anthropic. The Claude 3 Model Family: Opus, Sonnet, Haiku. Anthropic Technical Report, 2024. URL https://www.anthropic.com/news/claude-3-family.

[54] Google Gemini Team. Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context. arXiv Preprint arXiv:2403.05530, 2024. URL https://arxiv.org/abs/2403.05530.

[55] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending Context Window of Large Language Models via Positional Interpolation. arXiv Preprint arXiv:2306.15595, 2023. URL https://arxiv.org/abs/2306.15595.

[56] Nelson F. Liu, Kevin Lin, John Hewitt, et al. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics, 2024. URL https://arxiv.org/abs/2307.03172.

[57] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer Feed-Forward Layers Are Key-Value Memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021.

[58] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer Normalization. arXiv Preprint arXiv:1607.06450, 2016. URL https://arxiv.org/abs/1607.06450.

[59] Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. In Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1910.07467.

[60] Meta AI. The Llama 4 Herd: The Beginning of a New Era of Natively Multimodal AI. Meta AI Blog, 2025. URL https://ai.meta.com/blog/llama-4-multimodal-intelligence/.

[61] Mistral AI. Mistral Large 2. Mistral AI Blog, 2024. URL https://mistral.ai/news/mistral-large-2407/.

[62] DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv Preprint arXiv:2412.19437, 2024. URL https://arxiv.org/abs/2412.19437.

[63] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient Streaming Language Models with Attention Sinks. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2309.17453.

[64] Yao Fu, Rameswar Panda, Xinyao Niu, et al. Data Engineering for Scaling Language Models to 128K Context. arXiv Preprint arXiv:2402.10171, 2024. URL https://arxiv.org/abs/2402.10171.

[65] Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2312.00752.

[66] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019. URL https://arxiv.org/abs/1905.09418.

[67] Catherine Olsson, Nelson Elhage, Neel Nanda, et al. In-Context Learning and Induction Heads. Transformer Circuits Thread, 2022. URL https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.

[68] Zhengbao Wu, Aman Arora, Zhiqiang Wang, Byung-Gon Kim, and Tian Huang. Retrieval Head Mechanistically Explains Long-Context Factuality. arXiv Preprint arXiv:2404.15574, 2024. URL https://arxiv.org/abs/2404.15574.

[69] Jesse Vig. A Multiscale Visualization of Attention in the Transformer Model. In Proceedings of the 57th ACL: System Demonstrations, 2019. URL https://arxiv.org/abs/1906.05714.

[70] Samira Abnar and Willem Zuidema. Quantifying Attention Flow in Transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020. URL https://arxiv.org/abs/2005.00928.

[71] Oren Barkan, Edan Hauon, Avi Caciularu, Ido Dagan, and Noam Koenigstein. Grad-SAM: Explaining Transformers via Gradient Self-Attention Maps. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM), 2021. URL https://arxiv.org/abs/2104.13299.

[72] Sarthak Jain and Byron C. Wallace. Attention Is Not Explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2019. URL https://arxiv.org/abs/1902.10186.

[73] Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse Autoencoders Find Highly Interpretable Features in Language Models. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2309.08600.

[74] Trenton Bricken, Adly Templeton, Joshua Batson, et al. Towards Monosemanticity: Decomposing Language Models with Dictionary Learning. Transformer Circuits Thread, 2023. URL https://transformer-circuits.pub/2023/monosemantic-features/index.html.

[75] Adly Templeton, Tom Conerly, Jonathan Marcus, et al. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Transformer Circuits Thread, 2024. URL https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html.

[76] Anthropic. Natural Language Autoencoders: Interpreting Neural Networks with Natural Language Descriptions. Anthropic Research Blog, 2026. URL https://www.anthropic.com/research/natural-language-autoencoders.

[77] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning Representations by Back-Propagating Errors. Nature, 1986. URL https://doi.org/10.1038/323533a0.

[78] Herbert Robbins and Sutton Monro. A Stochastic Approximation Method. The Annals of Mathematical Statistics, 1951.

[79] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations (ICLR), 2015. URL https://arxiv.org/abs/1412.6980.

[80] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. arXiv Preprint arXiv:1711.05101, 2019. URL https://arxiv.org/abs/1711.05101.

[81] Shengding Hu, Yuge Tu, Xu Han, et al. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. arXiv Preprint arXiv:2404.06395, 2024. URL https://arxiv.org/abs/2404.06395.

[82] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2307.08691.

[83] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision. arXiv Preprint arXiv:2407.08691, 2024. URL https://arxiv.org/abs/2407.08691.

[84] Ted Zadouri, Jay Shah, Ganesh Bikshandi, and Tri Dao. FlashAttention-4: Hardware-Efficient Attention on Blackwell GPUs with Minimal Software Design. arXiv Preprint arXiv:2603.05451, 2026. URL https://arxiv.org/abs/2603.05451.

[85] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. Training Compute-Optimal Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2203.15556.

[86] Katherine Lee, Daphne Ippolito, Andrew Nystrom, et al. Deduplicating Training Data Makes Language Models Better. In Proceedings of the 60th Annual Meeting of the ACL, 2022. URL https://arxiv.org/abs/2107.06499.

[87] Chunting Zhou, Pengfei Liu, Puxin Xu, et al. LIMA: Less Is More for Alignment. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2305.11206.

[88] Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, et al. Liger-Kernel: Efficient Triton Kernels for LLM Training. arXiv Preprint arXiv:2410.10989, 2024. URL https://arxiv.org/abs/2410.10989.

[89] Daniel Han and Michael Han. Unsloth: Efficient LLM Fine-Tuning, 2024. URL https://github.com/unslothai/unsloth.

[90] PyTorch Team. Torchtune: PyTorch Native Post-Training Library, 2024. URL https://github.com/pytorch/torchtune.

[91] Neel Jain, Ping yeh Chiang, Yuxin Wen, et al. NEFTune: Noisy Embeddings Improve Instruction Finetuning. In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2310.05914.

[92] Edward J. Hu, Yelong Shen, Phillip Wallis, et al. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR), 2022. URL https://arxiv.org/abs/2106.09685.

[93] Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021. URL https://arxiv.org/abs/2012.13255.

[94] Damjan Kalajdzievski. A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. arXiv Preprint arXiv:2312.03732, 2023. URL https://arxiv.org/abs/2312.03732.

[95] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient Finetuning of Quantized Language Models. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2305.14314.

[96] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, et al. DoRA: Weight-Decomposed Low-Rank Adaptation. arXiv Preprint arXiv:2402.09353, 2024. URL https://arxiv.org/abs/2402.09353.

[97] Soufiane Hayou, Nikhil Ghosh, and Bin Yu. LoRA+: Efficient Low Rank Adaptation of Large Models. arXiv Preprint arXiv:2402.12354, 2024. URL https://arxiv.org/abs/2402.12354.

[98] Qingru Zhang, Minshuo Chen, Alexander Bukharin, et al. AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. arXiv Preprint arXiv:2303.10512, 2023. URL https://arxiv.org/abs/2303.10512.

[99] Dawid Jan Kopiczko, Tijmen Blankevoort, and Markus Nagel. VeRA: Vector-Based Random Matrix Adaptation. arXiv Preprint arXiv:2310.11454, 2024. URL https://arxiv.org/abs/2310.11454.

[100] Neil Houlsby, Andrei Giber, Stanislaw Jastrzebski, et al. Parameter-Efficient Transfer Learning for NLP. In International Conference on Machine Learning (ICML), 2019. URL https://arxiv.org/abs/1902.00751.

[101] Xiang Lisa Li and Percy Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Proceedings of the 59th Annual Meeting of the ACL, 2021. URL https://arxiv.org/abs/2101.00190.

[102] Brian Lester, Rami Al-Rfou, and Noah Constant. The Power of Scale for Parameter-Efficient Prompt Tuning. In Proceedings of the 2021 Conference on EMNLP, 2021. URL https://arxiv.org/abs/2104.08691.

[103] Haokun Liu, Derek Tam, Mohammed Muqeeth, et al. Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper Than in-Context Learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2205.05638.

[104] Elad Ben Zaken, Shauli Ravfogel, and Yoav Goldberg. BitFit: Simple Parameter-Efficient Fine-Tuning for Transformer-Based Masked Language-Models. In Proceedings of the 60th Annual Meeting of the ACL, 2022. URL https://arxiv.org/abs/2106.10199.

[105] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, et al. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In International Conference on Learning Representations (ICLR), 2017. URL https://arxiv.org/abs/1701.06538.

[106] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, et al. Mixtral of Experts. arXiv Preprint arXiv:2401.04088, 2024. URL https://arxiv.org/abs/2401.04088.

[107] Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with Gumbel-Softmax. In International Conference on Learning Representations (ICLR), 2017.

[108] DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv Preprint arXiv:2405.04434, 2024. URL https://arxiv.org/abs/2405.04434.

[109] William Fedus, Barret Zoph, and Noam Shazeer. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. Journal of Machine Learning Research, 2022.

[110] Databricks. DBRX: A New State-of-the-Art Open LLM. Databricks Blog, 2024. URL https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm.

[111] Jiayi Zhang, Simon Yu, Derek Chong, et al. Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity. arXiv Preprint arXiv:2510.01171, 2025. URL https://arxiv.org/abs/2510.01171.

[112] Ashwin K. Vijayakumar, Michael Cogswell, Ramprasaath R. Selvaraju, et al. Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models. AAAI, 2018.

[113] Minh Nguyen. Min-p Sampling: A Simple Baseline for Better LLM Decoding. arXiv Preprint arXiv:2310.06022, 2024.

[114] Xian Li, Ari Holtzman, Daniel Fried, et al. Contrastive Decoding: Open-Ended Text Generation as Optimization. ACL, 2023.

[115] Brandon T. Willard and Rémi Louf. Efficient Guided Generation for Large Language Models. arXiv Preprint arXiv:2307.09702, 2023. URL https://arxiv.org/abs/2307.09702.

[116] Yixin Dong, Charlie Moon, Yuekai Wang, et al. XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models. arXiv Preprint arXiv:2411.15100, 2024.

[117] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An Explanation of in-Context Learning as Implicit Bayesian Inference. In Proceedings of the 10th International Conference on Learning Representations (ICLR), 2022. URL https://arxiv.org/abs/2111.02080.

[118] Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. Function Vectors in Large Language Models. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2310.15213.

[119] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), 2022. URL https://arxiv.org/abs/2104.08786.

[120] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What Makes Good in-Context Examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO), ACL Workshop, 2022. URL https://arxiv.org/abs/2101.06804.

[121] Sewon Min, Xinxi Lyu, Ari Holtzman, et al. Rethinking the Role of Demonstrations: What Makes in-Context Learning Work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022. URL https://arxiv.org/abs/2202.12837.

[122] Jason Wei, Xuezhi Wang, Dale Schuurmans, et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2201.11903.

[123] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large Language Models Are Zero-Shot Reasoners. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2205.11916.

[124] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2203.11171.

[125] Shunyu Yao, Dian Yu, Jeffrey Zhao, et al. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2305.10601.

[126] Lei Wang, Wanyu Xu, Yiber Lan, et al. Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023. URL https://arxiv.org/abs/2305.04091.

[127] Shunyu Yao, Jeffrey Zhao, Dian Yu, et al. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2210.03629.

[128] Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv.org/abs/2005.11401.

[129] Yuntao Bai, Andy Jones, Kamal Ndousse, et al. Constitutional AI: Harmlessness from AI Feedback. arXiv Preprint arXiv:2212.08073, 2022. URL https://arxiv.org/abs/2212.08073.

[130] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, et al. Large Language Models Are Human-Level Prompt Engineers. In Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2211.01910.

[131] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, et al. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2310.03714.

[132] Chengrun Yang, Xuezhi Wang, Yifeng Lu, et al. Large Language Models as Optimizers. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2309.03409.

[133] Jingwen Yang, Yuxuan Zhu, Binyuan Wang, et al. ARQ: Attentive Reasoning Queries for Multi-Hop Question Answering over Long Contexts. arXiv Preprint arXiv:2501.08290, 2025. URL https://arxiv.org/abs/2501.08290.

[134] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. arXiv Preprint arXiv:2210.17323, 2023. URL https://arxiv.org/abs/2210.17323.

[135] Ji Lin, Jiaming Tang, Haotian Tang, et al. AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration. In Proceedings of Machine Learning and Systems (MLSys), 2024. URL https://arxiv.org/abs/2306.00978.

[136] Georgi Gerganov. GGUF: GPT-Generated Unified Format (llama.cpp), 2023. URL https://github.com/ggerganov/llama.cpp.

[137] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. URL https://arxiv.org/abs/2211.10438.

[138] Zechun Liu, Barlas Oguz, Changsheng Zhao, et al. LLM-QAT: Data-Free Quantization Aware Training for Large Language Models. arXiv Preprint arXiv:2305.17888, 2023. URL https://arxiv.org/abs/2305.17888.

[139] Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babber, and Dan Alistarh. Extreme Compression of Large Language Models via Additive Quantization. arXiv Preprint arXiv:2401.06118, 2024. URL https://arxiv.org/abs/2401.06118.

[140] Elias Frantar and Dan Alistarh. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. URL https://arxiv.org/abs/2301.00774.

[141] Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. A Simple and Effective Pruning Approach for Large Language Models. In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2306.11695.

[142] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network. arXiv Preprint arXiv:1503.02531, 2015.

[143] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast Inference from Transformers via Speculative Decoding. In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. URL https://arxiv.org/abs/2211.17192.

[144] Tianle Cai, Yuhong Li, Zhengyang Geng, et al. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2401.10774.

[145] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2401.15077.

[146] Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the Sequential Dependency of LLM Inference Using Lookahead Decoding. arXiv Preprint arXiv:2402.02057, 2024. URL https://arxiv.org/abs/2402.02057.

[147] Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. Better & Faster Large Language Models via Multi-Token Prediction. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2404.19737.

[148] Ziwei Ji, Nayeon Lee, Rita Frieske, et al. Survey of Hallucination in Natural Language Generation. ACM Computing Surveys, 2023. URL https://arxiv.org/abs/2202.03629.

[149] Saurav Kadavath, Tom Conerly, Amanda Askell, et al. Language Models (Mostly) Know What They Know. arXiv Preprint arXiv:2207.05221, 2022. URL https://arxiv.org/abs/2207.05221.

[150] Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. In Proceedings of EMNLP, 2023. URL https://arxiv.org/abs/2303.08896.

[151] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. In International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2302.09664.

[152] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. DoLA: Decoding by Contrasting Layers Improves Factuality in Large Language Models. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2309.03883.

[153] Isabel O. Gallegos, Ryan A. Rossi, Joe Barber, Shanshan Tong, et al. Bias and Fairness in Large Language Models: A Survey. Computational Linguistics, 2024. URL https://arxiv.org/abs/2309.00770.

[154] Nicholas Carlini, Florian Tramer, Eric Wallace, et al. Extracting Training Data from Large Language Models. USENIX Security Symposium, 2021. URL https://arxiv.org/abs/2012.07805.

[155] Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv Preprint arXiv:2307.15043, 2023. URL https://arxiv.org/abs/2307.15043.

[156] Ethan Perez, Saffron Huang, Francis Song, et al. Red Teaming Language Models with Language Models. In Proceedings of EMNLP, 2022. URL https://arxiv.org/abs/2202.03286.

[157] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP), 2023. URL https://arxiv.org/abs/2309.06180.

[158] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction, 2018. URL http://incompleteideas.net/book/the-book-2nd.html.

[159] Richard S Sutton. Learning to Predict by the Methods of Temporal Differences. Machine Learning, 1988.

[160] Christopher J. C. H Watkins. Learning from Delayed Rewards. 1989.

[161] Gavin A. Rummery and Mahesan Niranjan. On-Line q-Learning Using Connectionist Systems, 1994.

[162] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al. Human-Level Control Through Deep Reinforcement Learning. Nature, 2015. URL https://www.nature.com/articles/nature14236.

[163] Long-Ji Lin. Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching. Machine Learning, 1992.

[164] Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized Experience Replay. In Proceedings of the 4th International Conference on Learning Representations (ICLR), 2016. URL https://arxiv.org/abs/1511.05952.

[165] Ronald J Williams. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning, 1992. URL https://link.springer.com/article/10.1007/BF00992696.

[166] Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, et al. Asynchronous Methods for Deep Reinforcement Learning. In Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016. URL https://arxiv.org/abs/1602.01783.

[167] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust Region Policy Optimization. In Proceedings of the 32nd International Conference on Machine Learning (ICML), 2015. URL https://arxiv.org/abs/1502.05477.

[168] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv Preprint arXiv:1707.06347, 2017. URL https://arxiv.org/abs/1707.06347.

[169] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. In Proceedings of the 4th International Conference on Learning Representations (ICLR), 2016. URL https://arxiv.org/abs/1506.02438.

[170] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. In Proceedings of the 35th International Conference on Machine Learning (ICML), 2018. URL https://arxiv.org/abs/1801.01290.

[171] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. Nature, 2020. URL https://arxiv.org/abs/1911.08265.

[172] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. In Proceedings of the 8th International Conference on Learning Representations (ICLR), 2020. URL https://arxiv.org/abs/1912.01603.

[173] Andrew Y. Ng, Daishi Harada, and Stuart J. Russell. Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. 收录于 *第 16 届国际机器学习大会(16th International Conference on Machine Learning, ICML)*，1999 年。

[174] Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, 等. Fine-Tuning Language Models from Human Preferences. arXiv 预印本 arXiv:1909.08593，2019 年。URL：https://arxiv.org/abs/1909.08593.

[175] Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, 与 Dario Amodei. Deep Reinforcement Learning from Human Preferences. 收录于 *神经信息处理系统进展(Advances in Neural Information Processing Systems, NeurIPS)*，2017 年。URL：https://arxiv.org/abs/1706.03741.

[176] Leandro von Werra, Younes Belkada, Lewis Tunstall, 等. TRL: Transformer Reinforcement Learning，2022 年。URL：https://github.com/huggingface/trl.

[177] Junkang Wang, Jianfei Duan, Yue Liu, Zhangchen Yue, Hanghang Tong, 与 James Wang. Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints. 收录于 *第 12 届国际学习表示大会(12th International Conference on Learning Representations, ICLR)*，2024 年。URL：https://arxiv.org/abs/2309.16240.

[178] Sayak Ray Chowdhury, Anush Chakraborty, Srinadh Natarajan, Alekh Agarwal, 与 David Sontag. Provably Robust DPO: Aligning Language Models with Noisy Feedback. arXiv 预印本 arXiv:2403.00409，2024 年。URL：https://arxiv.org/abs/2403.00409.

[179] Alexander Gorbatenko. Online DPO with Synchronised Reference Model Updates. TRL 文档，2024 年。

[180] Jiatao Ji, Adam Fisch, Jason Weston, 与 Sainbayar Sukhbaatar. Towards Exact Optimization of Language Model Alignment. arXiv 预印本 arXiv:2402.05369，2024 年。URL：https://arxiv.org/abs/2402.05369.

[181] Huayu Chen, Guande Zheng, Yimeng Kim, 与 Yiwen Chen. Noise Contrastive Alignment of Language Models with Explicit Rewards. arXiv 预印本 arXiv:2402.05369，2024 年。URL：https://arxiv.org/abs/2402.05369.

[182] Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, 与 Peter J. Liu. SLiC-HF: Sequence Likelihood Calibration with Human Feedback. arXiv 预印本 arXiv:2305.10425，2023 年。URL：https://arxiv.org/abs/2305.10425.

[183] Yu Meng, Mengzhou Xia, 与 Danqi Chen. SimPO: Simple Preference Optimization with a Reference-Free Reward. 收录于 *神经信息处理系统进展(NeurIPS)*，2024 年。URL：https://arxiv.org/abs/2405.14734.

[184] Qiying Yu, Zheng Sun, Shang Wen, 等. DAPO: An Open-Source LLM Reinforcement Learning System. arXiv 预印本 arXiv:2503.14476，2025 年。URL：https://arxiv.org/abs/2503.14476.

[185] Zhiyu Chen, Yiwei Deng, Ruiqi Zhang, Hao Sun, 与 Weizhu Chen. GSPO: Sequence-Level Policy Optimization for Language Model Alignment. arXiv 预印本 arXiv:2502.12459，2025 年。URL：https://arxiv.org/abs/2502.12459.

[186] Yihao Liu, Lefan Han, Yifan Tan, 等. Understanding and Mitigating the Pretraining Distribution Bias in GRPO. arXiv 预印本 arXiv:2505.07888，2025 年。URL：https://arxiv.org/abs/2505.07888.

[187] Haoxiang Xu, Hongyuan Zhao, Ying Liu, 与 Dong Wei. It Takes Two: Pairwise Preference Optimization with Two Rollouts. arXiv 预印本 arXiv:2505.07856，2025 年。URL：https://arxiv.org/abs/2505.07856.

[188] Chi Han, Minlie Li, 与 Wenting Chen. SAPO: Soft Adaptive Policy Optimization for Efficient LLM Alignment. arXiv 预印本 arXiv:2503.01739，2025 年。URL：https://arxiv.org/abs/2503.01739.

[189] Yanqi Zhong, Yifu Chen, Zijun Li, 与 Minlie Chen. Importance Sampling Corrections for Large Language Model Alignment with Asynchronous Generation. arXiv 预印本 arXiv:2503.09057，2025 年。URL：https://arxiv.org/abs/2503.09057.

[190] Zhixun Luo, Jiaqi Shi, Tao Yu, 与 Minlie Chen. VESPO: Variational Sequence-Level Soft Policy Optimization for LLM Alignment. arXiv 预印本 arXiv:2505.07508，2025 年。URL：https://arxiv.org/abs/2505.07508.

[191] Yanxu An, Li Shen, Yifan Xu, 与 Xinmei Liu. DPPO: Direct Divergence-Based Policy Optimization for Language Model Alignment. arXiv 预印本 arXiv:2503.14532，2025 年。URL：https://arxiv.org/abs/2503.14532.

[192] Zhenyu Luo, Ziyan Chen, Yulun Jiang, 等. ScaleRL: Scaling Reinforcement Learning for LLM Reasoning. arXiv 预印本 arXiv:2505.16356，2025 年。URL：https://arxiv.org/abs/2505.16356.

[193] Yanqi Zhong, Jiaqi Shi, Yifu Chen, 与 Minlie Chen. GDPO: Learning to Directly Align Language Models with Group-Decoupled Reward. arXiv 预印本 arXiv:2501.17888，2025 年。URL：https://arxiv.org/abs/2501.17888.

[194] Seokhyun Choi, Hyunji Park, Doyoung Moon, Kyuyoung Kim, 与 Edward Choi. GOPO: Group Ordinal Policy Optimization for LLM Alignment with Non-Verifiable Rewards. arXiv 预印本 arXiv:2505.12948，2025 年。URL：https://arxiv.org/abs/2505.12948.

[195] Shangmin Guo, Biao Zhang, Tianlin Liu, 等. Direct Language Model Alignment from Online AI Feedback. arXiv 预印本 arXiv:2402.04792，2024 年。URL：https://arxiv.org/abs/2402.04792.

[196] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, 等. WebGPT: Browser-Assisted Question-Answering with Human Feedback. arXiv 预印本 arXiv:2112.09332，2021 年。URL：https://arxiv.org/abs/2112.09332.

[197] Leo Gao, John Schulman, 与 Jacob Hilton. Scaling Laws for Reward Model Overoptimization. 收录于 *第 40 届国际机器学习大会(ICML)*，2023 年。

[198] Ralph Allan Bradley 与 Milton E. Terry. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. *Biometrika*，1952 年。URL：https://www.jstor.org/stable/2334029.

[199] R. L Plackett. The Analysis of Permutations. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*，1975 年。

[200] Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, 与 Hang Li. Listwise Approach to Learning to Rank: Theory and Algorithm. 收录于 *第 25 届国际机器学习大会(ICML)*，2008 年。

[201] Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, 与 Hang Li. Learning to Rank: From Pairwise Approach to Listwise Approach. 收录于 *第 24 届国际机器学习大会(ICML)*，2007 年。

[202] Christopher J. C. Burges, Robert Ragno, 与 Quoc V. Le. Learning to Rank with Nonsmooth Cost Functions. 收录于 *神经信息处理系统进展(NeurIPS)*，2006 年。

[203] Chris Burges, Tal Shaked, Erin Renshaw, 等. Learning to Rank Using Gradient Descent. 收录于 *第 22 届国际机器学习大会(ICML)*，2005 年。

[204] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, 等. Overcoming Catastrophic Forgetting in Neural Networks. 收录于 *美国国家科学院院刊(Proceedings of the National Academy of Sciences)*，2017 年。

[205] Shen Li, Yanli Zhao, Rohan Varma, 等. PyTorch Distributed: Experiences on Accelerating Data Parallel Training. 收录于 *VLDB Endowment*，2020 年。

[206] Alexander Sergeev 与 Mike Del Balso. Horovod: Fast and Easy Distributed Deep Learning in TensorFlow. arXiv 预印本 arXiv:1802.05799，2018 年。

[207] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, 与 Bryan Catanzaro. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv 预印本 arXiv:1909.08053，2019 年。

[208] Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, 等. Reducing Activation Recomputation in Large Transformer Models. 收录于 *机器学习与系统大会(Proceedings of Machine Learning and Systems, MLSys)*，2023 年。

[209] Yanping Huang, Youlong Cheng, Ankur Bapna, 等. GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism. 收录于 *神经信息处理系统进展(NeurIPS)*，2019 年。

[210] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, 等. PipeDream: Generalized Pipeline Parallelism for DNN Training. 收录于 *第 27 届 ACM 操作系统原理研讨会(27th ACM Symposium on Operating Systems Principles, SOSP)*，2019 年。

[211] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, 等. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. arXiv 预印本 arXiv:2104.04473，2021 年。

[212] Penghui Qi, Xinyi Wan, Guangxing Huang, 与 Min Lin. Zero Bubble Pipeline Parallelism. arXiv 预印本 arXiv:2401.10241，2023 年。

[213] Samyam Rajbhandari, Jeff Rasley, Olatunji Rber, 与 Yuxiong He. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. arXiv 预印本 arXiv:1910.02054，2020 年。

[214] Yanli Zhao, Andrew Gu, Rohan Varma, 等. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. 收录于 *VLDB Endowment*，2023 年。

[215] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, 与 Byung-Gon Chun. Orca: A Distributed Serving System for Transformer-Based Generative Models. 收录于 *第 16 届 USENI 操作系统设计与实现大会(16th USENIX Symposium on Operating Systems Design and Implementation, OSDI)*，2022 年。

[216] Zhewei Yao, Samyam Rajbhandari, Reza Yazdani Aminabadi, 等. DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-Like Models at All Scales. arXiv 预印本 arXiv:2308.01320，2023 年。

[217] Jian Hu, Xibin Tao, Weixun Zhu, Sicheng Yang, Jingwen Liu, 与 Zilin Li. OpenRLHF: An Easy-to-Use, Scalable and High-Performance RLHF Framework. arXiv 预印本 arXiv:2405.11143，2024 年。

[218] Tianqi Chen, Bing Xu, Chiyuan Zhang, 与 Carlos Guestrin. Training Deep Nets with Sublinear Memory Cost. arXiv 预印本 arXiv:1604.06174，2016 年。

[219] Paulius Micikevicius, Sharan Narang, Jonah Alben, 等. Mixed Precision Training. 收录于 *国际学习表示大会(ICLR)*，2018 年。

[220] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, 与 Yuxiong He. ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning. arXiv 预印本 arXiv:2104.07857，2021 年。

[221] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, 等. PaLM: Scaling Language Modeling with Pathways. arXiv 预印本 arXiv:2204.02311，2022 年。

[222] Sam McCandlish, Jared Kaplan, Dario Amodei, 与 OpenAI Dota Team. An Empirical Model of Large-Batch Training. arXiv 预印本 arXiv:1812.06162，2018 年。

[223] Eric Zelikman, Yuhuai Wu, Jesse Mu, 与 Noah D. Goodman. STaR: Bootstrapping Reasoning with Reasoning. 收录于 *神经信息处理系统进展(NeurIPS)*，2022 年。URL：https://arxiv.org/abs/2203.14465.

[224] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, 与 Shunyu Yao. Reflexion: Language Agents with Verbal Reinforcement Learning. 收录于 *神经信息处理系统进展(NeurIPS)*，2023 年。URL：https://arxiv.org/abs/2303.11366.

[225] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, 与 Yu-Xiong Wang. Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models. arXiv 预印本 arXiv:2310.04406，2024 年。

[226] Pranav Putta, Edmund Mills, Naman Garg, 等. Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents. arXiv 预印本 arXiv:2408.07199，2024 年。

[227] Xingyao Wang, Boxuan Ding, Ziniu Hoang, 等. OpenHands: An Open Platform for AI Software Developers as Generalist Agents. arXiv 预印本 arXiv:2407.16741，2024 年。

[228] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, 等. Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv 预印本 arXiv:2305.16291，2023 年。URL：https://arxiv.org/abs/2305.16291.

[229] Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, 与 Steven Hoi. CodeRL: Mastering Code Generation Through Pretrained Models and Deep Reinforcement Learning. 收录于 *神经信息处理系统进展(NeurIPS)*，2023 年。

[230] Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, 与 Noah D. Goodman. Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking. arXiv 预印本 arXiv:2403.09629，2024 年。URL：https://arxiv.org/abs/2403.09629.

[231] Arian Hosseini, Xingdi Yuan, Pascal Poupart, Adam Trischler, 与 Yoshua Bengio. V-STaR: Training Verifiers for Self-Taught Reasoners. arXiv 预印本 arXiv:2402.06457，2024 年。

[232] John Yang, Carlos E. Jimenez, Alexander Wettig, 等. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. 收录于 *神经信息处理系统进展(NeurIPS)*，2024 年。URL：https://arxiv.org/abs/2405.15793.

[233] Xiao Yu, Baolin Peng, Ruize Xu, 等. Reinforcement World Model Learning for LLM-Based Agents. arXiv 预印本 arXiv:2602.05842，2026 年。

[234] Shunyu Yao, Dian Yu, Jeffrey Zhao, 等. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. 收录于 *神经信息处理系统进展(NeurIPS)*，2024 年。URL：https://arxiv.org/abs/2305.10601.

[235] Maciej Besta, Nils Blach, Ales Kubicek, 等. Graph of Thoughts: Solving Elaborate Problems with Large Language Models. 收录于 *AAAI 人工智能大会(AAAI Conference on Artificial Intelligence)*，2024 年。

[236] Nisan Stiennon, Long Ouyang, Jeffrey Wu, 等. Learning to Summarize from Human Feedback. 收录于 *神经信息处理系统进展(NeurIPS)*，2020 年。URL：https://arxiv.org/abs/2009.01325.

[237] Levente Kocsis 与 Csaba Szepesvári. Bandit Based Monte-Carlo Planning. 收录于 *欧洲机器学习大会(European Conference on Machine Learning, ECML)*，2006 年。

[238] Google DeepMind. AlphaProof and AlphaGeometry 2: Solving Olympiad Geometry Without Human Demonstrations，2024 年。URL：https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/.

[239] Zhenting Qi, Mingyuan Wan, Jialin Cao, 与 Min Lin. Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers. arXiv 预印本 arXiv:2408.06195，2024 年。

[240] Aman Madaan, Niket Tandon, Prakhar Gupta, 等. Self-Refine: Iterative Refinement with Self-Feedback. 收录于 *神经信息处理系统进展(NeurIPS)*，2023 年。

[241] OpenAI. Learning to Reason with LLMs，2024 年。URL：https://openai.com/index/learning-to-reason-with-llms/.

[242] OpenAI. OpenAI o3 and o4-mini System Card，2025 年。URL：https://openai.com/index/o3-and-o4-mini-system-card/.

[243] Hunter Lightman, Vineet Kosaraju, Yura Burda, 等. Let's Verify Step by Step. arXiv 预印本 arXiv:2305.20050，2023 年。

[244] Qwen Team. QwQ: Reflect Deeply on the Boundaries of the Unknown. Qwen 博客，2024 年。URL：https://qwenlm.github.io/blog/qwq-32b-preview/.

[245] Qwen Team. Qwen3 Technical Report. arXiv 预印本 arXiv:2505.09388，2025 年。

[246] Peiyi Wang, Lei Li, Zhihong Shao, 等. Math-Shepherd: Verify and Reinforce LLMs Step-by-Step Without Human Annotations. arXiv 预印本 arXiv:2312.08935，2024 年。URL：https://arxiv.org/abs/2312.08935.

[247] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, 等. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv 预印本 arXiv:2411.15124，2024 年。URL：https://arxiv.org/abs/2411.15124.

[248] Yiwei Qin, Xuefeng Li, Haoyang Zou, 等. O1 Replication Journey: A Strategic Progress Report. arXiv 预印本 arXiv:2410.18982，2024 年。URL：https://arxiv.org/abs/2410.18982.

[249] Charlie Snell, Jaehoon Lee, Kelvin Xu, 与 Aviral Kumar. Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters. arXiv 预印本 arXiv:2408.03314，2024 年。

[250] Zhenyu Wu, Qinghua Hu, Yin Zhang, Yiming Gao, 与 Jiangtao Chen. An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models. arXiv 预印本 arXiv:2408.00724，2024 年。

[251] Jared Kaplan, Sam McCandlish, Tom Henighan, 等. Scaling Laws for Neural Language Models. arXiv 预印本 arXiv:2001.08361，2020 年。URL：https://arxiv.org/abs/2001.08361.

[252] Jacob Cohen. A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*，1960 年。

[253] Joseph L Fleiss. Measuring Nominal Scale Agreement Among Many Raters. *Psychological Bulletin*，1971 年。

[254] Chuan Guo, Geoff Pleiss, Yu Sun, 与 Kilian Q. Weinberger. On Calibration of Modern Neural Networks. 收录于 *国际机器学习大会(ICML)*，2017 年。

[255] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, 等. Self-Instruct: Aligning Language Models with Self-Generated Instructions. arXiv 预印本 arXiv:2212.10560，2022 年。URL：https://arxiv.org/abs/2212.10560.

[256] Can Xu, Qingfeng Sun, Kai Zheng, 等. WizardLM: Empowering Large Language Models to Follow Complex Instructions. arXiv 预印本 arXiv:2304.12244，2023 年。URL：https://arxiv.org/abs/2304.12244.

[257] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, 等. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. 收录于 *神经信息处理系统进展(NeurIPS)*，2023 年。URL：https://arxiv.org/abs/2306.05685.

[258] Arpad E Elo. The Rating of Chess Players, Past and Present，1978 年。

[259] Ralf Herbrich, Tom Minka, 与 Thore Graepel. TrueSkill™: A Bayesian Skill Rating System. 收录于 *神经信息处理系统进展(NeurIPS)*，2006 年。URL：https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html.

[260] Edwin B Wilson. Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association*，1927 年。URL：https://www.jstor.org/stable/2276774.

[261] Kishore Papineni, Salim Roukos, Todd Ward, 与 Wei-Jing Zhu. BLEU: A Method for Automatic Evaluation of Machine Translation. 收录于 *第 40 届计算语言学协会年会(40th Annual Meeting of the Association for Computational Linguistics, ACL)*，2002 年。URL：https://aclanthology.org/P02-1040/.

[262] Chin-Yew Lin. ROUGE: A Package for Automatic Evaluation of Summaries. 收录于 *文本摘要分支：ACL-04 工作坊(Text Summarization Branches Out: Proceedings of the ACL-04 Workshop)*，2004 年。URL：https://aclanthology.org/W04-1013/.

[263] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, 与 Yoav Artzi. BERTScore: Evaluating Text Generation with BERT. 收录于 *国际学习表示大会(ICLR)*，2020 年。URL：https://arxiv.org/abs/1904.09675.

[264] Satanjeev Banerjee 与 Alon Lavie. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. 收录于 *ACL 机器翻译与/或摘要的内在与外在评价度量大数工作坊*，2005 年。URL：https://aclanthology.org/W05-0909/.

[265] Mark Chen, Jerry Tworek, Heewoo Jun, 等. Evaluating Large Language Models Trained on Code. arXiv 预印本 arXiv:2107.03374，2021 年。URL：https://arxiv.org/abs/2107.03374.

[266] Carlos E. Jimenez, John Yang, Alexander Wettig, 等. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? 收录于 *国际学习表示大会(ICLR)*，2024 年。URL：https://arxiv.org/abs/2310.06770.

[267] Shuyan Zhou, Frank F. Xu, Hao Zhu, 等. WebArena: A Realistic Web Environment for Building Autonomous Agents. 收录于 *国际学习表示大会(ICLR)*，2024 年。URL：https://arxiv.org/abs/2307.13854.

[268] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, 与 Matthew Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. 收录于 *国际学习表示大会(ICLR)*，2021 年。

[269] Xiao Liu, Hao Yu, Hanchen Zhang, 等. AgentBench: Evaluating LLMs as Agents. arXiv 预印本 arXiv:2308.03688，2023 年。

[270] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, 与 Chenguang Zhu. G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment. 收录于 *2023 年自然语言处理实证方法大会(2023 Conference on Empirical Methods in Natural Language Processing, EMNLP)*，2023 年。URL：https://arxiv.org/abs/2303.16634.

[271] Dan Hendrycks, Collin Burns, Steven Basart, 等. Measuring Massive Multitask Language Understanding. 收录于 *国际学习表示大会(ICLR)*，2021 年。

[272] Charles A. E Goodhart. Problems of Monetary Management: The U.K. Experience. *Monetary Theory and Practice*，1984 年。

[273] Stephen Robertson 与 Hugo Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*，2009 年。

[274] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, 等. Dense Passage Retrieval for Open-Domain Question Answering. 收录于 *2020 年自然语言处理实证方法大会(EMNLP)*，2020 年。URL：https://arxiv.org/abs/2004.04906.

[275] Jeff Johnson, Matthijs Douze, 与 Hervé Jégou. Billion-Scale Similarity Search with GPUs. *IEEE Transactions on Big Data*，2021 年。

[276] Yu. A. Malkov 与 D. A. Yashunin. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*，2020 年。

[277] Gordon V. Cormack, Charles L. A. Clarke, 与 Stefan Buettcher. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. 收录于 *第 32 届国际 ACM SIGIR 大会*，2009 年。

[278] Thibault Formal, Benjamin Piwowarski, 与 Stéphane Clinchant. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. 收录于 *第 44 届国际 ACM SIGIR 信息检索研究与发展大会*，2021 年。

[279] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, 与 Stéphane Clinchant. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. arXiv 预印本 arXiv:2109.10086，2021 年。

[280] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, 与 Jimmy Lin. Document Ranking with a Pretrained Sequence-to-Sequence Model. *EMNLP Findings*，2020 年。

[281] Payal Bajaj, Daniel Campos, Nick Craswell, 等. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. arXiv 预印本 arXiv:1611.09268，2016 年。

[282] Omar Khattab 与 Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. 收录于 *第 43 届国际 ACM SIGIR 大会*，2020 年。URL：https://arxiv.org/abs/2004.12832.

[283] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, 与 Matei Zaharia. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. 收录于 *2022 年计算语言学协会北美分会大会(2022 Conference of the North American Chapter of the Association for Computational Linguistics, NAACL)*，2022 年。

[284] Karen Sparck Jones. A Statistical Interpretation of Term Specificity and Its Application in Retrieval. *Journal of Documentation*，1972 年。

[285] Rodrigo Nogueira 与 Kyunghyun Cho. Passage Re-Ranking with BERT. arXiv 预印本 arXiv:1901.04085，2019 年。

[286] Luyu Gao, Xueguang Ma, Jimmy Lin, 与 Jamie Callan. Precise Zero-Shot Dense Retrieval Without Relevance Labels. 收录于 *第 61 届计算语言学协会年会(ACL)*，2023 年。URL：https://arxiv.org/abs/2212.10496.

[287] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, 与 Hannaneh Hajishirzi. Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection. arXiv 预印本 arXiv:2310.11511，2023 年。URL：https://arxiv.org/abs/2310.11511.

[288] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, 与 Zhen-Hua Ling. Corrective Retrieval Augmented Generation. arXiv 预印本 arXiv:2401.15884，2024 年。URL：https://arxiv.org/abs/2401.15884.

[289] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, 与 Jong C. Park. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models Through Question Complexity. arXiv 预印本 arXiv:2403.14403，2024 年。URL：https://arxiv.org/abs/2403.14403.

[290] Darren Edge, Ha Trinh, Newman Cheng, 等. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv 预印本 arXiv:2404.16130，2024 年。URL：https://arxiv.org/abs/2404.16130.

[291] Adrian H Rackauckas. RAG-Fusion: A New Take on Retrieval-Augmented Generation，2024 年。

[292] Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, 与 Vijai Mohan. REFRAG: Rethinking RAG Based Decoding. arXiv 预印本 arXiv:2509.01092，2025 年。

[293] Bowen Jin, Hansi Zeng, 等. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. arXiv 预印本 arXiv:2503.09516，2025 年。

[294] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, 等. Natural Questions: A Benchmark for Question Answering Research. *Transactions of the Association for Computational Linguistics*，2019 年。

[295] Mandar Joshi, Eunsol Choi, Daniel Weld, 与 Luke Zettlemoyer. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. 收录于 *ACL*，2017 年。

[296] Zhilin Yang, Peng Qi, Saizheng Zhang, 等. HotpotQA: A Dataset for Diverse, Explainable Multi-Hop Question Answering. 收录于 *EMNLP*，2018 年。

[297] Shahul Es, Jithin James, Luis Espinosa-Anke, 与 Steven Schockaert. RAGAs: Automated Evaluation of Retrieval Augmented Generation. arXiv 预印本 arXiv:2309.15217，2023 年。URL：https://arxiv.org/abs/2309.15217.

[298] Nelson F. Liu, Kevin Lin, John Hewitt, 等. Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics*，2024 年。URL：https://arxiv.org/abs/2307.03172.

[299] Chankyu Lee, Rajarshi Roy, Menber Xu, 等. NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. arXiv 预印本 arXiv:2405.17428，2024 年。

[300] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, 与 Meishan Zhang. Towards General Text Embeddings with Multi-Stage Contrastive Learning. arXiv 预印本 arXiv:2308.03281，2023 年。

[301] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, 与 Zheng Liu. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. arXiv 预印本 arXiv:2402.03216，2024 年。

[302] Shitao Xiao, Zheng Liu, Peitian Zhang, 与 Niklas Muennighoff. C-Pack: Packaged Resources to Advance General Chinese Embedding. arXiv 预印本 arXiv:2309.07597，2023 年。

[303] Tianjun Zhang, Shishir G. Patil, Naman Jain, 等. RAFT: Adapting Language Model to Domain Specific RAG. arXiv 预印本 arXiv:2403.10131，2024 年。URL：https://arxiv.org/abs/2403.10131.

[304] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, 与 Ming-Wei Chang. REALM: Retrieval-Augmented Language Model Pre-Training. 收录于 *第 37 届国际机器学习大会(ICML)*，2020 年。URL：https://arxiv.org/abs/2002.08909.

[305] Endel Tulving. Memory and Consciousness. *Canadian Psychology / Psychologie Canadienne*，1985 年。

[306] Larry R Squire. Declarative and Nondeclarative Memory: Multiple Brain Systems Supporting Learning and Memory. *Journal of Cognitive Neuroscience*，1992 年。URL：https://doi.org/10.1162/jocn.1992.4.3.232.

[307] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, 等. Show Your Work: Scratchpads for Intermediate Computation with Language Models. arXiv 预印本 arXiv:2112.00114，2021 年。URL：https://arxiv.org/abs/2112.00114.

[308] Ruiqi Guo, Philip Sun, Erik Lindgren, 等. Accelerating Large-Scale Inference with Anisotropic Vector Quantization. 收录于 *国际机器学习大会(ICML)*，2020 年。

[309] Jianlv Chen, Shitao Luo, Mingzheng Zhang, Zheng Liu, Yingxia Xiao, 与 Defu Han. Hybrid Retrieval for Open-Domain Question Answering. arXiv 预印本 arXiv:2210.06029，2022 年。URL：https://arxiv.org/abs/2210.06029.

[310] Tiago Forte. Building a Second Brain: A Proven Method to Organize Your Digital Life and Unlock Your Creative Potential，2022 年。

[311] Steve Harris 与 Andy Seaborne. SPARQL 1.1 Query Language，2013 年。URL：https://www.w3.org/TR/sparql11-query/.

[312] Nadime Francis, Alastair Green, Paolo Guagliardo, 等. Cypher: An Evolving Query Language for Property Graphs. 收录于 *2018 年数据管理国际大会(2018 International Conference on Management of Data, SIGMOD)*，2018 年。

[313] Timothée Lacroix, Guillaume Obozinski, 与 Nicolas Usunier. Tensor Decompositions for Temporal Knowledge Base Completion. 收录于 *国际学习表示大会(ICLR)*，2020 年。URL：https://arxiv.org/abs/2004.04926.

[314] Jason Weston, Sumit Chopra, 与 Antoine Bordes. Memory Networks. 收录于 *国际学习表示大会(ICLR)*，2015 年。URL：https://arxiv.org/abs/1410.3916.

[315] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, 与 Rob Fergus. End-to-End Memory Networks. 收录于 *神经信息处理系统进展(NeurIPS)*，2015 年。URL：https://arxiv.org/abs/1503.08895.

[316] Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, 与 Joseph E. Gonzalez. MemGPT: Towards LLMs as Operating Systems. arXiv 预印本 arXiv:2310.08560，2023 年。URL：https://arxiv.org/abs/2310.08560.

[317] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, 与 Yongfeng Zhang. A-Mem: Agentic Memory for LLM Agents. 收录于 *神经信息处理系统进展(NeurIPS)*，2025 年。

[318] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, 与 Deshraj Yadav. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv 预印本 arXiv:2504.19413，2025 年。URL：https://arxiv.org/abs/2504.19413.

[319] Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, 与 Michael S. Bernstein. Generative Agents: Interactive Simulacra of Human Behavior. 收录于 *第 36 届 ACM 用户界面软件与技术年度研讨会(36th Annual ACM Symposium on User Interface Software and Technology, UIST)*，2023 年。URL：https://arxiv.org/abs/2304.03442.

[320] Hermann Ebbinghaus. Über Das Gedächtnis: Untersuchungen Zur Experimentellen Psychologie，1885 年。

[321] Barbara Hayes-Roth. A Blackboard Architecture for Control. *Artificial Intelligence*，1985 年。URL：https://doi.org/10.1016/0004-3702(85)90063-3.

[322] Marcin Andrychowicz, Filip Wolski, Alex Ray, 等. Hindsight Experience Replay. 收录于 *神经信息处理系统进展(NeurIPS)*，2017 年。

[323] Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, 与 Pieter Abbeel. RL2: Fast Reinforcement Learning via Slow Reinforcement Learning. arXiv 预印本 arXiv:1611.02779，2016 年。

[324] Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, 与 Trevor Darrell. Curiosity-Driven Exploration by Self-Supervised Prediction. 收录于 *第 34 届国际机器学习大会(ICML)*，2017 年。

[325] Alex Graves, Greg Wayne, Malcolm Reynolds, 等. Hybrid Computing Using a Neural Network with Dynamic External Memory. *Nature*，2016 年。

[326] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, 与 Dong Yu. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. 收录于 *国际学习表示大会(ICLR)*，2025 年。URL：https://arxiv.org/abs/2410.10813.

[327] Adyasha Maharana, Dong-Ho Lee, Sergey Tuber, Mohit Ruber, Francesco Barbieri, 与 Mohit Bansal. LoCoMo: Long-Context Conversation with Memory Operations. 收录于 *2024 年自然语言处理实证方法大会(EMNLP)*，2024 年。

[328] Xinrong Zhang, Yingfa Chen, Shengding Hu, 等. InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens. 收录于 *第 62 届计算语言学协会年会(ACL)*，2024 年。URL：https://arxiv.org/abs/2402.13718.

[329] Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, 与 Thomas L. Griffiths. Cognitive Architectures for Language Agents. *Transactions on Machine Learning Research (TMLR)*，2024 年。URL：https://arxiv.org/abs/2309.02427.

[330] Kevin Lin, Charlie Snell, Yu Wang, 等. Sleep-Time Compute: Beyond Inference Scaling at Test-Time. arXiv 预印本 arXiv:2504.13171，2025 年。URL：https://arxiv.org/abs/2504.13171.

[331] Alex L. Zhang, Seyyed Hasan Mahdavi, Percy Liang, 与 Tatsunori Hashimoto. Recursive Language Models. arXiv 预印本 arXiv:2512.24601，2025 年。URL：https://arxiv.org/abs/2512.24601.

[332] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, 等. Toolformer: Language Models Can Teach Themselves to Use Tools. 收录于 *神经信息处理系统进展(NeurIPS)*，2023 年。

[333] Shishir G. Patil, Tianjun Zhang, Xin Wang, 与 Joseph E. Gonzalez. Gorilla: Large Language Model Connected with Massive APIs. 收录于 *神经信息处理系统进展(NeurIPS)*，2024 年。URL：https://arxiv.org/abs/2305.15334.

[334] Yujia Qin, Shihao Liang, Yining Ye, 等. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs. 收录于 *第 12 届国际学习表示大会(ICLR)*，2024 年。URL：https://arxiv.org/abs/2307.16789.

[335] Anthropic. Model Context Protocol，2024 年。URL：https://modelcontextprotocol.io.

[336] OpenAI. Swarm: An Educational Framework for Lightweight Multi-Agent Orchestration，2024 年。URL：https://github.com/openai/swarm.

[337] LangChain Inc. LangGraph: Build Stateful Multi-Actor Applications with LLMs，2024 年。URL：https://github.com/langchain-ai/langgraph.

[338] Qingyun Wu, Gagan Bansal, Jieyu Zhang, 等. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv 预印本 arXiv:2308.08155，2023 年。

[339] Lingjiao Chen, Matei Zaharia, 与 James Zou. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv 预印本 arXiv:2305.05176，2023 年。URL：https://arxiv.org/abs/2305.05176.

[340] Harrison Chase. LangChain，2022 年。URL：https://github.com/langchain-ai/langchain.

[341] João Moura. CrewAI: Framework for Orchestrating Role-Playing Autonomous AI Agents，2023 年。URL：https://github.com/crewAIInc/crewAI.

[342] Anthropic. Building Effective Agents，2024 年。URL：https://www.anthropic.com/research/building-effective-agents.

[343] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, 等. Self-Consistency Improves Chain of Thought Reasoning in Language Models. 收录于 *国际学习表示大会(ICLR)*，2023 年。URL：https://arxiv.org/abs/2203.11171.

[344] Greg Brockman, Vicki Cheung, Ludwig Pettersson, 等. OpenAI Gym，2016 年。

[345] Jaeseok Yoo 与 Dongmin Shin. Adaptive Episode Length Adjustment for Multi-Agent Reinforcement Learning. 收录于 *第 24 届自治智能体与多智能体系统国际大会(24th International Conference on Autonomous Agents and Multiagent Systems, AAMAS)*，2025 年。

[346] Zhuokai Liu, Hao Dong, 等. DLER: Doing Length Penalty Right—Incentivizing More Intelligence Per Token. arXiv 预印本 arXiv:2510.15110，2025 年。URL：https://arxiv.org/abs/2510.15110.

[347] Qiguang Liu 等. Answer Convergence as a Signal for Early Stopping in Reasoning. 收录于 *EMNLP*，2025 年。URL：https://aclanthology.org/2025.emnlp-main.904/.

[348] Jiangfei Mei 等. APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-Tail Generation. arXiv 预印本 arXiv:2509.18521，2025 年。URL：https://arxiv.org/abs/2509.18521.

[349] Qinghao Hu, Shang Yang, 等. Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter. arXiv 预印本 arXiv:2511.16665，2025 年。URL：https://arxiv.org/abs/2511.16665.

[350] Minqi Jiang, Edward Grefenstette, 与 Tim Rocktäschel. Prioritized Level Replay. 收录于 *第 38 届国际机器学习大会(ICML)*，2021 年。URL：https://arxiv.org/abs/2010.03934.

[351] Michael Dennis, Natasha Jaques, Eugene Vinitsky, 等. Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design. 收录于 *神经信息处理系统进展(NeurIPS)*，2020 年。

[352] Yifan Wang 等. Improving Data Efficiency for LLM Reinforcement Fine-Tuning via Difficulty-Targeted Online Data Selection. 收录于 *神经信息处理系统进展(NeurIPS)*，2025 年。URL：https://arxiv.org/abs/2506.05316.

[353] Xingyu Liu 等. Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning. arXiv 预印本 arXiv:2505.08364，2025 年。URL：https://arxiv.org/abs/2505.08364.

[354] Jing Yu Koh, Robert Lo, Lawrence Jang, 等. VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks. 收录于 *第 62 届计算语言学协会年会(ACL)*，2024 年。URL：https://arxiv.org/abs/2401.13649.

[355] Xiang Deng, Yu Gu, Boyuan Zheng, et al. Mind2Web: Towards a Generalist Agent for the Web. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2306.06070.

[356] Tianbao Xie, Danyang Zhang, Jixuan Chen, et al. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. In Advances in Neural Information Processing Systems, 2024.

[357] Rogerio Bonatti, Dan Zhao, Francesco Bonacci, et al. Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale. arXiv Preprint arXiv:2409.08264, 2024. URL https://arxiv.org/abs/2409.08264.

[358] Jakub Lála, Odhran O'Donoghue, Aleksandar Shtedritski, Sam Cox, Samuel G. Rodriques, and Andrew D. White. PaperQA: Retrieval-Augmented Generative Agent for Scientific Research. arXiv Preprint arXiv:2312.07559, 2023. URL https://arxiv.org/abs/2312.07559.

[359] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv Preprint arXiv:2408.06292, 2024. URL https://arxiv.org/abs/2408.06292.

[360] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2310.03302.

[361] Heinrich Küttler, Nantas Nardelli, Alexander H. Miller, et al. The NetHack Learning Environment. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv.org/abs/2006.13760.

[362] Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: A Benchmark for General AI Assistants. In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2311.12983.

[363] Mike Lewis, Denis Yarats, Yann N. Dauphin, Devi Parikh, and Dhruv Batra. Deal or No Deal? End-to-End Learning for Negotiation Dialogues. In Proceedings of EMNLP, 2017.

[364] Kushal Chawla, Jaysa Ramirez, Rene Clever, Gale Lucas, Jonathan May, and Jonathan Gratch. CaSiNo: A Corpus of Campsite Negotiation Dialogues for Automatic Negotiation Systems. In Proceedings of NAACL, 2021.

[365] Sirui Hong, Mingchen Zhuge, Jonathan Chen, et al. MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework, 2024.

[366] Hugging Face. OpenEnv: An Interface Library for RL Post-Training with Environments, 2025. URL https://github.com/huggingface/OpenEnv.

[367] Mark Towers, Ariel Kwiatkowski, Jordan Terry, et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. NeurIPS Datasets and Benchmarks, 2024. URL https://arxiv.org/abs/2407.17032.

[368] Zhiheng Xi, Yiwen Ding, Wenxiang Chen, et al. AgentGym: Evolving Large Language Model-Based Agents Across Diverse Environments. arXiv Preprint arXiv:2406.04151, 2024. URL https://arxiv.org/abs/2406.04151.

[369] Thibault Le Sellier De Chezelles, Maxime Gasse, Alexandre Drouin, Massimo Caccia, et al. The BrowserGym Ecosystem for Web Agent Research. arXiv Preprint arXiv:2412.05467, 2024. URL https://arxiv.org/abs/2412.05467.

[370] Meta PyTorch Team. TorchForge: PyTorch-Native Post-Training at Scale, 2025. URL https://github.com/meta-pytorch/torchforge.

[371] JSON-RPC Working Group. JSON-RPC 2.0 Specification, 2010. URL https://www.jsonrpc.org/specification.

[372] Google. Agent2Agent (A2A) Protocol, 2025. URL https://developers.google.com/agent2agent.

[373] Tom Preston-Werner. Semantic Versioning 2.0.0, 2024. URL https://semver.org/.

[374] Reid G Smith. The Contract Net Protocol: High-Level Communication and Control in a Distributed Problem Solver. IEEE Transactions on Computers, 1980. URL https://doi.org/10.1109/TC.1980.1675516.

[375] Michael Jones, John Bradley, and Nat Sakimura. JSON Web Token (JWT), 2015. URL https://datatracker.ietf.org/doc/html/rfc7519.

[376] Brian Campbell, John Bradley, Nat Sakimura, and Torsten Lodderstedt. OAuth 2.0 Mutual-TLS Client Authentication and Certificate-Bound Access Tokens, 2020. URL https://datatracker.ietf.org/doc/html/rfc8705.

[377] World Wide Web Consortium. Decentralized Identifiers (DIDs) v1.0, 2022. URL https://www.w3.org/TR/did-core/.

[378] Eric Rescorla. The Transport Layer Security (TLS) Protocol Version 1.3, 2018. URL https://datatracker.ietf.org/doc/html/rfc8446.

[379] Dick Hardt. The OAuth 2.0 Authorization Framework, 2012. URL https://datatracker.ietf.org/doc/html/rfc6749.

[380] Gerhard Weiss. Multiagent Systems: A Modern Approach to Distributed Artificial Intelligence, 1999.

[381] Michael Wooldridge. An Introduction to MultiAgent Systems, 2009.

[382] Edmund H. Durfee, Victor R. Lesser, and Daniel D. Corkill. Trends in Cooperative Distributed Problem Solving. IEEE Transactions on Knowledge and Data Engineering, 1989. URL https://doi.org/10.1109/69.43404.

[383] Edward de Bono. Six Thinking Hats, 1985.

[384] Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving Factuality and Reasoning in Language Models Through Multiagent Debate. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2023. URL https://arxiv.org/abs/2305.14325.

[385] Lloyd S Shapley. Stochastic Games. In Proceedings of the National Academy of Sciences, 1953. URL https://www.pnas.org/doi/10.1073/pnas.39.10.1095.

[386] Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. In Advances in Neural Information Processing Systems (NeurIPS), 2017. URL https://arxiv.org/abs/1706.02275.

[387] Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning. In Proceedings of the 35th International Conference on Machine Learning (ICML), 2018. URL https://arxiv.org/abs/1803.11605.

[388] Sainbayar Sukhbaatar, Arthur Szlam, and Rob Fergus. Learning Multiagent Communication with Backpropagation. In Advances in Neural Information Processing Systems (NeurIPS), 2016. URL https://arxiv.org/abs/1605.07736.

[389] Abhishek Das, Théophile Gerber, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. TarMAC: Targeted Multi-Agent Communication. In Proceedings of the 36th International Conference on Machine Learning (ICML), 2019. URL https://arxiv.org/abs/1810.11187.

[390] Angeliki Lazaridou and Marco Baroni. Emergent Multi-Agent Communication in the Deep Learning Era. arXiv Preprint arXiv:2006.02419, 2020. URL https://arxiv.org/abs/2006.02419.

[391] Max Jaderberg, Wojciech M. Czarnecki, Iain Dunning, et al. Human-Level Performance in 3D Multiplayer Games with Population-Based Reinforcement Learning. Science, 2019. URL https://www.science.org/doi/10.1126/science.aau6249.

[392] Yoav Shoham and Kevin Leyton-Brown. Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations, 2008. URL http://www.masfoundations.org/.

[393] Kaiqing Zhang, Zhuoran Yang, and Tamer Başar. Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms. Handbook of Reinforcement Learning and Control, 2021. URL https://arxiv.org/abs/1911.10635.

[394] Noam Nisan, Tim Roughgarden, Éva Tardos, and Vijay V. Vazirani. Algorithmic Game Theory, 2007. URL https://www.cs.cmu.edu/~sandholm/cs15-892F13/algorithmic-game-theory.pdf.

[395] OpenAI. OpenAI Agents SDK, 2025. URL https://github.com/openai/openai-agents-python.

[396] Microsoft. Semantic Kernel: SDK for Integrating AI Models into Applications, 2023. URL https://github.com/microsoft/semantic-kernel.

[397] Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Prompting Is Programming: A Query Language for Large Language Models. In Proceedings of PLDI, 2023.

[398] Raja Parasuraman and Victor Riley. Humans and Automation: Use, Misuse, Disuse, Abuse. Human Factors, 1997. URL https://doi.org/10.1518/001872097778543886.

[399] Vercel. Vercel AI SDK, 2024. URL https://sdk.vercel.ai.

[400] Chainlit. Chainlit: Build Production-Ready Conversational AI Applications, 2024. URL https://chainlit.io.

[401] Abubakar Abid, Ali Abdalla, Ali Abid, Dawood Khan, Abdulrahman Alfozan, and James Zou. Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild, 2019. URL https://arxiv.org/abs/1906.02569.

[402] Streamlit Inc. Streamlit: The Fastest Way to Build and Share Data Apps, 2024. URL https://streamlit.io.

[403] LangChain Inc. LangGraph Studio: The First Agent IDE, 2024. URL https://github.com/langchain-ai/langgraph-studio.

[404] Javier García and Fernando Fernández. A Comprehensive Survey on Safe Reinforcement Learning. Journal of Machine Learning Research, 2015.

[405] Geoffrey Irving, Paul Christiano, and Dario Amodei. AI Safety via Debate. arXiv Preprint arXiv:1805.00899, 2018. URL https://arxiv.org/abs/1805.00899.

[406] Evan Hubinger, Carson Denison, Jesse Mu, et al. Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training. arXiv Preprint arXiv:2401.05566, 2024.

[407] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete Problems in AI Safety. arXiv Preprint arXiv:1606.06565, 2016. URL https://arxiv.org/abs/1606.06565.

[408] Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not What You've Signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. In Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec), 2023.

[409] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and Characterizing Reward Hacking. In Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/2209.13085.

[410] Seungone Kim, Se June Jang, et al. Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. Findings of the ACL, 2023. URL https://arxiv.org/abs/2305.02301.
