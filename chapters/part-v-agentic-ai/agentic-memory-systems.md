# 第 17 章 智能体记忆系统

## 17.1 动机：为什么智能体需要记忆

大语言模型（Large Language Model, LLM）在本质上是**无状态**的函数近似器：给定一个提示（prompt）$x$，它们会产生一个关于后续内容的分布 $p_\theta(y \mid x)$。每一次推理调用都是从零开始的。上下文窗口（context window）——模型能够关注到的有限词元（token）序列——是生成时唯一可用的信息。对于短小、自包含的任务来说，这已经足够。但对于长时程（long-horizon）的智能体任务而言，这是一个根本性的瓶颈。

### 上下文窗口瓶颈

令 $L$ 表示最大上下文长度（例如 GPT-4o 的 $L = 128{,}000$ 个 token）。一个 token 大致编码 4 个字符；一本典型的书大约有 50 万词，约为 67 万个 token。即使不考虑成本，一个连续工作数日的自主智能体（autonomous agent）所累积的观察、工具输出和推理轨迹，也无法容纳在任何固定大小的窗口中。记忆系统就是对这一物理约束的工程化应对。

当智能体缺乏持久记忆时，会出现三种不同的失效模式：

1. **上下文的灾难性遗忘（Catastrophic forgetting）。** 一旦某个事件滑出上下文窗口，它就不可挽回地丢失了。智能体无法回溯一万 token 之前做出的某个决定。
2. **无法从经验中学习。** 没有情景存储，每一次回合（episode）对智能体而言都是全新的。成功的策略无法复用，错误会被一再重复。
3. **缺乏个性化。** 用户偏好、领域事实和关系历史必须在每一次会话中重新建立，这会降低用户体验与效率。

### 作为认知架构的记忆

认知科学在生物智能体中区分了若干记忆系统 [305, 306]：**工作记忆**（working memory，对信息的主动操作）、**情景记忆**（episodic memory，自传体事件）、**语义记忆**（semantic memory，关于世界的知识）以及**过程记忆**（procedural memory，技能与习惯）。有效的智能体 AI（Agentic AI）系统也能从类似的区分中受益——这不是因为我们要模拟神经科学，而是因为这些类别反映了真正不同的访问模式、更新频率与检索机制。

形式上，我们将智能体建模为一个元组 $A = (\pi_\theta, M, R, W)$，其中 $\pi_\theta$ 是策略（即 LLM），$M$ 是记忆存储，$R: Q \times M \to D$ 是检索函数（将查询映射到检索到的文档），$W: M \times E \to M$ 是写入函数（用新经验 $E$ 更新记忆）。在每一步 $t$，智能体观察 $o_t$，检索相关上下文 $c_t = R(o_t, M)$，并采取行动：

$$
a_t \sim \pi_\theta\left(\cdot \mid [s_t;\; c_t;\; h_t]\right),
$$

其中 $s_t$ 是当前系统提示（system prompt），$c_t$ 是检索到的记忆，$h_t$ 是近期的上下文内历史。行动之后，智能体可能写入新信息：$M \leftarrow W(M, (o_t, a_t, r_t))$。

![图 17.1：智能体记忆系统的四分类法，对应认知科学的区分。每种记忆类型具有不同的访问模式、更新频率和检索机制。](images/part-v-agentic-ai/agentic-memory-systems/agentic-memory-systems-p321-01.png)

## 17.2 记忆类型分类

### 17.2.1 工作记忆（短期）

工作记忆是智能体的活跃工作区：当前正在被操作的信息。在 LLM 智能体中，它对应于：

- **草稿板（Scratchpads）。** 在给出最终答案之前，写入专门缓冲区的中间推理步骤（例如思维链 [122]、草稿板 [307]）。
- **思维链缓冲区。** 在答案 token $a$ 之前生成的推理 token 序列 $z_1, z_2, \dots, z_k$，建模为 $p(a \mid x) = \sum_{z} p(a \mid x, z)\, p(z \mid x)$。
- **对话上下文。** 保留在上下文窗口中的近期轮次历史 $[(u_1, a_1), \dots, (u_t, a_t)]$。

工作记忆速度快（零检索延迟——它已在上下文中）、易失（上下文清空即丢失）、且容量受限（以 $L$ 为上界）。

### 17.2.2 情景记忆（基于经验）

情景记忆存储按上下文和时间索引的特定过往事件。对于智能体而言：

- **过往交互。** 之前对话、任务尝试及其结果的完整或摘要记录。
- **成功轨迹。** 高奖励的动作序列，可作为相似未来任务的少样本（few-shot）范例被检索。
- **失败案例。** 附带根因标注的已记录错误，使智能体能避免重蹈覆辙。
- **检索增强的情景回忆。** 给定一个新任务 $q$，检索 $k$ 个最相似的过往回合 $\{e_i\}_{i=1}^{k}$ 并将其纳入上下文。

情景记忆通常实现为向量存储（见 17.3.1 节），对回合摘要建立嵌入（embedding）。

### 17.2.3 语义记忆（世界知识）

语义记忆编码与特定情景解耦的一般事实与概念：

- **事实性知识。** 实体、属性与关系（例如「巴黎是法国首都」）。
- **领域概念。** 与智能体任务领域相关的定义、分类体系与本体的定义。
- **知识图谱。** 结构化表示 $G = (V, E)$，其中节点 $v \in V$ 是实体，边 $e \in E$ 是带类型的关系。

与情景记忆不同，语义记忆与上下文无关：「水在 100°C 沸腾」这一事实，无论何时何地学到都成立。

### 17.2.4 过程记忆（技能）

过程记忆编码「如何做事」——已被自动化的技能与动作模式：

- **习得的工具使用模式。** 对哪个任务调用哪个 API、如何格式化输入、如何处理错误。
- **动作序列。** 多步骤流程（例如「部署代码：运行测试 → 构建镜像 → 推送 → 更新清单」）。
- **作为记忆的策略。** 模型权重 $\theta$ 本身就编码了过程性知识；在成功轨迹上做微调（fine-tuning）就是一种过程记忆的固化。

**记忆类型分类示例**

一个协助软件开发的智能体使用：

- **工作记忆：** 当前正在编辑的文件、刚收到的错误消息。
- **情景记忆：**「上周我在模块 X 中通过在第 42 行加空值检查修复了一个类似的 NullPointerException。」
- **语义记忆：**「Python 的 `asyncio.gather` 并发运行协程；除非 `return_exceptions=True`，否则异常会向上传播。」
- **过程记忆：** 标准的调试工作流：复现 → 隔离 → 假设 → 测试 → 修复。

## 17.3 记忆架构

### 17.3.1 基于 RAG 的记忆

检索增强生成（Retrieval-Augmented Generation, RAG）[128] 是 LLM 智能体外部记忆的主导范式。记忆存储 $M$ 是一个文档集合 $\{d_i\}_{i=1}^{N}$；检索将查询 $q$ 映射到一个有序子集。

**嵌入存储与向量数据库。** 每个文档 $d_i$ 由一个嵌入模型 $\phi$ 编码：$v_i = \phi(d_i) \in \mathbb{R}^{D}$。查询同样被编码：$\mathbf{q} = \phi(q)$。检索返回按相似度排序的前 $k$ 个文档：

$$
\text{Retrieve}(q, M, k) = \mathop{\arg\max}_{S \subseteq [N],\, |S|=k} \sum_{i \in S} \mathrm{sim}(\mathbf{q}, v_i),
$$

其中 $\mathrm{sim}(\cdot, \cdot)$ 通常是余弦相似度。近似最近邻（Approximate Nearest-Neighbor, ANN）索引（FAISS [275]、HNSW [276]、ScaNN [308]）使这一过程在 $N \sim 10^7$ 规模下仍可行。

**检索策略。**

- **稠密检索（Dense retrieval）。** 查询与文档都由神经编码器编码（例如 DPR [274]、text-embedding-3-large）。能捕捉语义相似性，但需要 GPU 推理。
- **稀疏检索（Sparse retrieval）。** 基于 token 重叠的 BM25 或 TF-IDF。速度快、可解释，对精确关键词匹配表现强。
- **混合检索（Hybrid retrieval）。** 通过倒数排名融合（Reciprocal Rank Fusion, RRF）结合稠密与稀疏得分：

$$
\mathrm{RRF}(d, k) = \sum_{r \in \text{rankers}} \frac{1}{k + \mathrm{rank}_r(d)},
$$

其中 $k = 60$ 是平滑常数。混合检索持续优于任一单独方法 [309]。

**重排序（Re-ranking）。** 一个交叉编码器（cross-encoder）重排器 $f_\psi(q, d) \in [0, 1]$ 对每个检索到的文档与查询联合打分，以 $O(k)$ 次前向传播为代价换取更高精度。流水线为：用 ANN 检索 $k' \gg k$ 个候选，用交叉编码器重排，返回前 $k$ 个。

> **检索幻觉风险**
>
> RAG 并不能消除幻觉——它反而可能引入幻觉。如果检索到的文档过时、错误或仅有表面相关性，模型可能自信地纳入虚假信息。务必附带溯源元数据（来源、时间戳、置信度），并考虑进行忠实性（faithfulness）验证步骤。

### 17.3.2 基于摘要的记忆

当逐字存储代价过高或噪声过大时，摘要会在存储前压缩信息。

**渐进式摘要（Progressive Summarization）。** 在每一步 $t$，智能体维护一个滚动摘要 $S_t$。当新信息 $e_t$ 到达时：

$$
S_{t+1} = \mathrm{LLM}\bigl(\text{"Summarize: } [S_t] + [e_t]\text{"}\bigr).
$$

这使记忆大小保持 $O(1)$，但有丢失细节的风险。

**分层压缩。** 将记忆组织为多个层级 $L_0 \supset L_1 \supset \cdots \supset L_K$，其中 $L_0$ 是逐字内容，每个 $L_{i+1}$ 是 $L_i$ 的摘要。检索先检查 $L_K$（压缩程度最高、最快），再按需向下钻取。这呼应了 Forte [310] 的渐进式摘要技术。

**何时摘要 vs. 逐字存储。**

- **逐字存储：** 精确的事实、代码片段、数值结果、用户原话。
- **摘要：** 叙事上下文、推理链、冗余观察。
- **丢弃：** 噪声、不含信息内容的失败工具调用。

### 17.3.3 基于图的记忆

**知识图谱。** 知识图谱 $G = (V, E, R)$ 以三元组 $(h, r, t)$ 存储事实，其中 $h, t \in V$ 是实体，$r \in R$ 是关系。智能体可通过 SPARQL [311]、Cypher [312] 或「自然语言到图」的翻译来查询。

**实体关系抽取。** 新观察由抽取模型 $\mathrm{IE}: \text{text} \to \{(h_i, r_i, t_i)\}$ 解析，并合并入 $G$。共指消解（coreference resolution）与实体链接（entity linking）保证一致性。

**GraphRAG。** GraphRAG [290] 通过图遍历来增强 RAG：给定查询，先检索种子实体，再通过 $k$ 跳邻域遍历进行扩展，以浮现未被嵌入相似度直接匹配的相关事实。这对多跳（multi-hop）推理尤其强大：

$$
\mathrm{GraphRetrieve}(q, G, k) = \bigcup_{v \in \mathrm{seeds}(q)} N_k(v, G),
$$

其中 $N_k(v, G)$ 是 $v$ 的 $k$ 跳邻域。

**时序知识图谱。** 事实具有有效区间：$(h, r, t, [t_{\text{start}}, t_{\text{end}}])$。时序知识图谱 [313] 支持诸如「2023 年 OpenAI 的 CEO 是谁？」这样的查询，而不会混淆过去与现在的状态。

### 17.3.4 键值记忆网络

可微记忆网络（differentiable memory networks）[314, 315] 将记忆表示为一组键值对 $\{(k_i, v_i)\}_{i=1}^{M}$，采用基于软注意力的检索：

$$
\alpha_i = \mathrm{softmax}\!\left(\frac{q^{\top} k_i}{\sqrt{D}}\right), \qquad
c = \sum_{i=1}^{M} \alpha_i v_i.
$$

检索到的上下文 $c$ 是查询的可微函数，从而支持端到端训练。现代 Transformer 的注意力机制就是这一机制的特例。在智能体应用中，记忆槽可通过梯度下降或显式写操作来更新。

### 17.3.5 MemGPT 与虚拟上下文管理

MemGPT [316] 引入了一种虚拟上下文抽象，类比于操作系统中的虚拟内存。记忆被组织为若干层级：

**换入 / 换出（Page-In / Page-Out）策略。** 智能体根据以下因素决定将哪些记忆提升到热上下文（page-in）、哪些驱逐（page-out）：

- **近期性（Recency）：** 最近访问过的项更可能被再次需要。
- **相关性（Relevance）：** 与当前查询相似度高的项。
- **重要性（Importance）：** 在写入时标记为高重要性的项。

**自驱动的记忆管理。** 在 MemGPT 中，LLM 本身将记忆管理函数调用（`memory_search`、`memory_insert`、`memory_delete`）作为其动作空间的一部分发出。这使得记忆管理成为一种习得的行为，而非硬编码的策略——这也是 RL 训练（见 17.7 节）的天然目标。

## 17.4 记忆操作

### 17.4.1 写入：提交到记忆

并非每一个观察都应当被存储。写入决策是一个过滤问题：

$$
\mathrm{Write}(e) = \mathbb{1}[\mathrm{importance}(e) > \tau],
$$

其中 $\tau$ 是阈值，$\mathrm{importance}(e)$ 可以是：

- **意外度（Surprise）：** $-\log p_\theta(e \mid \text{context})$——出乎意料的事件信息量更大。
- **奖励信号：** 与高 $|r_t|$（正或负）相关的事件值得记住。
- **LLM 自评估：** 提示模型在 1–10 的尺度上对重要性打分。

**矛盾检测。** 在写入新事实 $f_{\text{new}}$ 之前，检查其与现有记忆的冲突：

$$
\mathrm{Conflict}(f_{\text{new}}, M) = \exists\, f \in M:\; \mathrm{Contradicts}(f_{\text{new}}, f).
$$

矛盾检测可通过自然语言推理（NLI）模型或提示 LLM 来实现。一旦发生冲突，智能体必须决定：覆盖、带时间戳保留两者，还是标记交由人工审核。

**记忆格式与粒度。** 除了「存什么」之外，「怎么存」同样至关重要。记忆条目从原子事实到冗长的逐字稿不等，各有不同的权衡：

**表 17.1：记忆粒度的权衡。**

| 格式 | 优点 | 缺点 |
|---|---|---|
| 原子事实（「用户偏好 Python。」） | 检索精确；可组合；易于去重与矛盾检测 | 丢失上下文；抽取易错；对细微信息较脆弱 |
| 结构化笔记（A-MEM [317]） | 元数据丰富（标签、链接）；支持图遍历；兼顾精度与上下文 | 写入成本高；需要模式设计 |
| 摘要式回合（MemGPT [316]） | 保留叙事连贯性；紧凑；适合多轮推理 | 摘要有损；难以局部更新 |
| 逐字稿 | 无损；无抽取误差；支持精确引用 | 存储占用大；检索噪声多；扫描代价高 |

在实践中，生产系统常常组合多种粒度 [318]：抽取原子事实用于精确召回，维护回合摘要用于叙事上下文，并将逐字稿归档到冷存储以备审计。Generative Agents 架构 [319] 将观察存储为带有自然语言描述、重要性评分和时间戳的原子「记忆对象」——既支持精确检索，也支持时序推理。

**设计准则。**

- **将粒度与查询类型匹配。** 如果用户问事实型问题（「我的 API key 是什么？」），原子事实最佳。如果问上下文型问题（「我们当初为什么决定用 Redis？」），就需要回合摘要。
- **在能力允许范围内以最细粒度存储，再在其上构建更粗的视图。** 对原子事实做摘要很容易；从有损摘要中还原原子几乎不可能。
- **附带溯源信息。** 每个记忆条目都应链接回其来源（对话轮次、文档、工具输出），以便智能体验证、用户审计。

### 17.4.2 读取 / 检索

**查询构造。** 检索查询 $q$ 不必是原始观察。更好的策略包括：

- **HyDE（假设性文档嵌入，Hypothetical Document Embeddings）[286]：** 先生成一个假设性答案，对其做嵌入，再以该嵌入作为查询。
- **查询扩展（Query expansion）：** 生成查询的多个改写版本，取检索结果的并集。
- **退一步提示（Step-back prompting）：** 在检索前将具体查询抽象为更一般的问题。

**时间衰减与近期偏置。** 较旧的记忆可能相关性更低。一种时间加权得分：

$$
\mathrm{score}(d, q, t) = \lambda \cdot \mathrm{sim}(\mathbf{q}, v_d) + (1 - \lambda) \cdot \exp\!\left(-\frac{t - t_d}{\tau_{\text{decay}}}\right),
$$

其中 $t_d$ 是记忆的创建时间，$\tau_{\text{decay}}$ 控制衰减速率。Generative Agents 论文 [319] 采用了类似的近期加权检索。

### 17.4.3 更新：冲突解决与固化

记忆固化（consolidation）合并相关记忆以降低冗余、浮现更高层的模式：

$$
M' = \mathrm{Consolidate}(M) = \mathrm{Cluster}(M) \cup \mathrm{Summarize}(\mathrm{Cluster}(M)).
$$

**遗忘机制。** 生物记忆会遗忘；人工记忆也应如此。策略包括：

- **LRU 驱逐：** 当容量超限时移除最近最少使用的条目。
- **重要性加权遗忘：** $p(\text{forget} \mid d) \propto \exp(-\mathrm{importance}(d))$。
- **间隔重复（Spaced repetition）：** 被反复访问的记忆保留更久，遵循指数遗忘曲线 [320]。

### 17.4.4 反思：元认知操作

反思（Reflection）[224, 319] 是一种高阶记忆操作：智能体读取自身的记忆并生成洞见：

$$
\mathrm{Reflect}(M) \to \{i_1, i_2, \dots\} \subset M_{\text{semantic}},
$$

其中每个洞见 $i_j$ 是从多个情景记忆中推导出的更高层抽象。

**实践中的反思（Reflexion）**

在三次尝试解决一个编程问题失败后，智能体进行反思：

1. 从情景记忆中检索三次失败的回合。
2. 生成一条洞见：「我总是忘记处理输入列表为空的边界情况。」
3. 将这条洞见存入语义记忆。
4. 在下一次尝试时，检索该洞见并显式地检查空输入。

这正是 Reflexion [224] 的核心机制：通过自我反思进行的言语强化学习（verbal reinforcement learning）。

**反思存放在哪里？** 反思从情景记忆中读取，但写入语义记忆。所得洞见是与上下文无关的泛化（「总是检查空输入」），而非特定于某一回合的记录——因此它们属于语义记忆 $M_{\text{semantic}}$。然而，在反思过程本身进行时，中间推理（被检索的回合 + 合成提示 + 生成的洞见）占据的是工作记忆（上下文窗口）。简言之：

- **输入：** 情景记忆（特定的过往事件）
- **计算：** 工作记忆（上下文中的主动推理）
- **输出：** 语义记忆（持久的、泛化的洞见）

这映射了生物记忆固化：在睡眠与反思期间，情景经验被逐步转化为语义知识。

## 17.5 多轮对话的记忆

### 17.5.1 用户建模与偏好追踪

一个持久的用户模型 $U$ 存储：

- **显式偏好：** 明确陈述的喜好/厌恶、沟通风格偏好。
- **隐式偏好：** 从行为中推断（例如用户总是要求用 Python 写代码、偏好简洁的回答）。
- **专业水平：** 从词汇与问题复杂度推断出的领域知识。
- **目标与上下文：** 正在进行的项目、当前任务、组织角色。

用户模型在每次交互后更新：

$$
U_{t+1} = \mathrm{Update}(U_t, (u_t, a_t, \text{feedback}_t)).
$$

### 17.5.2 会话连续性

没有记忆，每次对话都是冷启动。有了会话记忆：

1. 在会话开始时，检索用户模型 $U$ 与最近的会话摘要。
2. 注入个性化系统提示：「你正在帮助 Alice，一位从事分布式训练项目的高级 ML 工程师。上次会话你帮她调试了一个梯度同步问题。」
3. 在会话结束时，总结该会话并更新 $U$。

### 17.5.3 通过记忆实现个性化

个性化同时提升效率（更少的澄清性问题）与质量（针对用户专业水平校准的回答）。关键技术包括：

- **自适应详略度：** 根据用户的历史参与度调整回答长度。
- **领域预热：** 从语义记忆中前置相关领域上下文。
- **主动召回：** 无需被要求就浮现相关的过往交互（「你上个月问过这个话题；这是我们当时找到的结果」）。

> **隐私与记忆**
>
> 持久的用户记忆引发重大的隐私关切。智能体必须：（1）在存储个人信息前获得明确同意；（2）提供检查与删除已存记忆的机制；（3）在多用户部署中执行访问控制；（4）遵守数据保留法规（GDPR、CCPA）。记忆系统应默认以隐私为先（privacy-by-default）来设计。

## 17.6 多智能体系统的记忆

当多个智能体协作完成共享任务时，记忆成为一种**协调机制**——而不仅仅是个人知识库。一个分解任务的规划智能体必须向执行者智能体传达子目标；一个评审智能体必须访问与其所评智能体相同的上下文；一个由智能体组成的研究团队必须避免重复劳动。没有共享记忆，智能体只能通过直接消息传递所有信息，造成带宽瓶颈，并在对话滑出上下文时丢失信息。共享记忆通过提供一个所有智能体都可读写的、持久的、可查询的底层介质来解决这一问题——将隐式协调（「希望另一个智能体还记得」）转化为显式状态（「答案在黑板（blackboard）上」）。

### 17.6.1 共享记忆池

在多智能体系统（Multi-Agent System）中，智能体可以在私有存储 $M_i$ 之外共享一个公共记忆存储 $M_{\text{shared}}$：

$$
\text{context}_i(t) = R(M_i, q_i) \cup R(M_{\text{shared}}, q_i).
$$

共享记忆支持隐式协调：智能体 A 写入一项发现；智能体 B 无需显式通信即可检索到它。

### 17.6.2 黑板架构

黑板模式 [321] 是一种经典的多智能体协调机制：每个智能体从黑板读写。一个控制器监控黑板，并在某个智能体的前置条件满足时激活它。这解耦了智能体：它们通过共享状态而非直接消息来通信。

### 17.6.3 共享知识中的一致性与冲突

当多个智能体向共享记忆写入时，冲突就会出现。解决策略包括：

- **后写者赢（Last-write-wins）：** 简单但会丢失信息。
- **版本化记忆：** 维护所有写入的历史；智能体可查询任意版本。
- **投票 / 共识：** 要求 $n$ 个智能体中有 $k$ 个同意后才能提交某事实。
- **置信度加权合并：** $f_{\text{merged}} = \sum_i w_i f_i$，其中 $w_i$ 是智能体 $i$ 的置信度。
- **指定权威：** 将记忆区域的归属权分配给特定智能体。

> **开放问题：分布式记忆一致性**
>
> 多智能体系统在并发写入、网络分区与对抗性智能体下应如何维持记忆一致性？经典的分布式系统方案（Paxos、Raft）适用但代价高昂。对于许多智能体任务而言，具有有界陈旧度（bounded staleness）的近似一致性可能就足够了——但恰当的权衡点仍是一个开放的研究问题。

## 17.7 用强化学习训练记忆系统

### 17.7.1 记忆操作的奖励信号

记忆操作（读、写、更新、反思）可被视为强化学习（Reinforcement Learning, RL）框架中的动作。挑战在于设计激励有用记忆行为的奖励信号：

- **任务奖励传播。** 如果某次记忆检索导致了正确答案，则将该奖励归功于那次检索动作。稀疏但明确。
- **检索精度奖励。** $r_{\text{retrieve}} = \mathrm{Relevance}(d_{\text{retrieved}}, \text{task})$，由一个习得的相关性模型估计。
- **记忆效率奖励。** 惩罚不必要的写入：$r_{\text{write}} = -\lambda \cdot \mathbb{1}[\text{write}]$，鼓励选择性存储。
- **一致性奖励。** 奖励内部一致（无矛盾）的记忆状态。

第 $t$ 步记忆操作 $m_t$ 的组合奖励：

$$
r^{\text{mem}}_t = r^{\text{task}}_t + \alpha \cdot r^{\text{retrieve}}_t + \beta \cdot r^{\text{write}}_t + \gamma \cdot r^{\text{consistency}}_t.
$$

### 17.7.2 学习该记住什么

「该记住什么」的问题是一个元学习（meta-learning）挑战：智能体必须学习一个写入策略 $\pi_{\text{write}}(e)$，以最大化未来任务表现。这很困难，因为：

1. 记忆的价值只在将来才显现（延迟奖励）。
2. 写入时可能的未来查询空间是未知的。
3. 记忆之间存在交互：存储 $e$ 的价值取决于 $M$ 中还有什么。

方法：

- **事后重标注（Hindsight relabeling）[322]。** 在一次成功回合之后，回顾性地将被检索到的记忆标注为「重要」，并训练写入策略存储类似条目。
- **元强化学习（Meta-RL）[323]。** 跨任务分布训练写入策略；策略学会存储跨任务可泛化的信息。
- **好奇心驱动的存储 [324]。** 存储令人意外（高预测误差）的观察，因为它们很可能信息量大。

### 17.7.3 记忆增强的策略优化

联合优化策略与其记忆系统的思想可追溯至可微记忆网络 [325]，并由 REALM [304] 扩展到检索增强的 LLM。一个记忆增强智能体的完整策略梯度目标：

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^{t} r_t\right] - \lambda \cdot \mathcal{L}_{\text{mem}}(\phi),
$$

其中 $\theta$ 是 LLM 参数，$\phi$ 是记忆系统参数（例如检索模型权重），$\mathcal{L}_{\text{mem}}$ 是关于记忆复杂度的正则项。

> **关键洞见：将记忆作为习得的归纳偏置**
>
> 用 RL 训练记忆操作使智能体能发展出针对特定任务的记忆策略。编程智能体学会存储 API 签名；研究智能体学会存储引用链；客服智能体学会存储用户投诉模式。记忆系统成为一种为智能体所在领域量身定制的、习得的归纳偏置（inductive bias）。

## 17.8 记忆方法对比

**表 17.2：跨关键维度对各类智能体记忆架构的对比。**

| 架构 | 容量 | 检索 | 更新成本 | 可训练 | 最适合 |
|---|---|---|---|---|---|
| 上下文内（工作记忆） | $O(L)$ token | 0 ms | 免费 | 通过微调 | 短任务、主动推理 |
| 稠密 RAG [128] | $O(10^7)$ 文档 | 10–50 ms | $O(1)$ 嵌入 | 仅编码器 | 语义搜索、问答 |
| 稀疏（BM25）[273] | $O(10^8)$ 文档 | 1–5 ms | $O(\|d\|)$ 索引 | 否 | 关键词搜索、法律/医疗 |
| 混合 RAG [309] | $O(10^7)$ 文档 | 15–60 ms | $O(1)$ 嵌入 | 仅编码器 | 通用检索 |
| 摘要 | 无限 | 0 ms（上下文内） | $O(\|e\|)$ LLM 调用 | 通过微调 | 长对话、叙事 |
| 知识图谱 [313] | $O(10^9)$ 三元组 | 5–100 ms | $O(1)$ 插入 | 嵌入层 | 结构化事实、多跳 |
| KV 记忆网络 [315] | $O(M)$ 槽 | $O(M)$ 注意力 | 梯度步 | 完全 | 端到端可微任务 |
| MemGPT 分层 [316] | 无限 | 0–100 ms | 混合 | 通过 RL | 长时程智能体、助手 |
| Graph RAG [290] | $O(10^7)$ 节点 | 20–200 ms | $O(1)$ 插入 | 仅编码器 | 复杂推理、社区 |

## 17.9 评估记忆系统

评估智能体记忆极具挑战，因为记忆操作的质量只能间接地通过长时程上的下游任务表现来揭示。一个对已存事实召回完美的记忆系统，仍可能因检索到无关上下文或淹没 LLM 的上下文窗口而失败。

### 17.9.1 评估维度

LongMemEval [326] 确立了一个长期记忆系统必须展现的五大核心能力：

1. **信息抽取。** 系统能否识别并存储对话轮次中的显著事实？用事实召回率度量：从记忆中可恢复的真实事实占多少比例。
2. **多会话推理。** 系统能否综合分散在多个过往会话中的信息？例如，「根据我们上周和昨天的对话，项目范围发生了哪些变化？」
3. **时序推理。** 系统能否正确回答依赖时间的查询？例如，「在重组之前我说我的优先事项是什么？」需要区分不同的时序状态。
4. **知识更新。** 当事实发生变化（用户搬家、偏好转移）时，记忆是否反映最新状态同时保留历史？
5. **拒答。** 当系统没有相关记忆时，是否正确地说「我不知道」，而不是幻觉出一个貌似合理但虚构的回忆？

### 17.9.2 基准

**表 17.3：用于评估智能体记忆系统的基准（benchmark）。**

| 基准 | 会议 | 规模 | 关注点 |
|---|---|---|---|
| LongMemEval [326] | ICLR 2025 | 500 个问题，可扩展的历史 | 五项记忆能力；多会话聊天 |
| LOCOMO [327] | EMNLP 2024 | 多会话对话 | 单跳、时序、多跳、开放域 QA |
| InfiniteBench [328] | ACL 2024 | 100K+ token 上下文 | 长上下文召回，非记忆专用但测试极限 |

### 17.9.3 指标

**记忆层指标。**

- **记忆召回率（Memory Recall）：** $\dfrac{\text{可从记忆中检索到的真实事实数}}{\text{真实事实总数}}$。度量存储的完整性。
- **记忆精确率（Memory Precision）：** $\dfrac{\text{前 } k \text{ 检索中相关条目数}}{k}$。度量检索中的噪声。
- **延迟（Latency）：** 从查询到检索到上下文的时间（p50 与 p95）。
- **token 效率：** 每次查询注入上下文的 token 总数。越低越好——无关上下文会降低 LLM 准确率并增加成本。

**下游指标。**

- **回答准确率：** 在记忆条件下的最终回答正确性（EM、F1，或 LLM-as-judge）。
- **忠实性（Faithfulness）：** 回答是否准确反映记忆内容，没有捏造？
- **个性化质量：** 用户满意度，通过偏好评分或「记忆增强 vs 无记忆」系统的 A/B 测试度量。
- **矛盾率：** 系统产出与先前所述事实不一致回答的频率。

**运维指标。**

- **写入选择性：** 触发记忆写入的轮次比例。过高 → 噪声；过低 → 缺口。
- **陈旧度（Staleness）：** 尽管存在更新却仍检索到过时事实的频率。
- **存储增长率：** 每交互小时存储的 token 数。无界增长不可持续。

> **评估鸿沟**
>
> 大多数记忆论文在短基准（10–50 个会话）上评估。真实的生产级智能体运行数月、数千个会话。长时程评估——其中记忆漂移、矛盾累积与存储膨胀成为主导失效模式——仍是一个开放挑战。从业者应将基准得分与运维指标的纵向监控相结合。

## 17.10 实现模式

### 17.10.1 带嵌入的向量存储记忆

最常见的记忆模式将条目存储为嵌入向量并附带元数据（时间戳、重要性评分、标签）。检索将余弦相似度与时间衰减结合，使近期且重要的记忆优先浮现。重复检测与 LRU 驱逐保持存储有界。

```python
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    content: str
    embedding: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    source: str = "agent"

class VectorMemoryStore:
    """
    Hybrid dense+sparse memory store with temporal decay.
    Supports importance-weighted retrieval and LRU eviction.
    """
    def __init__(
        self,
        embed_fn,               # callable: str -> np.ndarray
        max_entries: int = 10_000,
        decay_rate: float = 0.01,   # per hour
        recency_weight: float = 0.3,
    ):
        self.embed_fn = embed_fn
        self.max_entries = max_entries
        self.decay_rate = decay_rate
        self.recency_weight = recency_weight
        self.entries: list[MemoryEntry] = []

    # -- Write --------------------------------------------------------------
    def write(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        check_duplicates: bool = True,
    ) -> MemoryEntry:
        """Commit a new memory, evicting if at capacity."""
        if check_duplicates and self._is_duplicate(content):
            return None  # Skip near-duplicate entries
        embedding = self.embed_fn(content)
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            importance=importance,
            tags=tags or [],
        )
        if len(self.entries) >= self.max_entries:
            self._evict()
        self.entries.append(entry)
        return entry

    def _is_duplicate(self, content: str, threshold: float = 0.95) -> bool:
        """Check if a near-duplicate already exists."""
        if not self.entries:
            return False
        emb = self.embed_fn(content)
        sims = self._cosine_similarities(emb)
        return float(np.max(sims)) > threshold

    def _evict(self):
        """Remove the least important + least recent entry."""
        now = datetime.now()
        scores = []
        for e in self.entries:
            age_hours = (now - e.timestamp).total_seconds() / 3600
            recency = np.exp(-self.decay_rate * age_hours)
            score = e.importance * (1 - self.recency_weight) \
                + recency * self.recency_weight
            scores.append(score)
        worst_idx = int(np.argmin(scores))
        self.entries.pop(worst_idx)

    # -- Retrieve -----------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int = 5,
        recency_boost: bool = True,
    ) -> list[MemoryEntry]:
        """
        Hybrid retrieval: dense similarity + temporal recency.
        Returns top-k entries sorted by combined score.
        """
        if not self.entries:
            return []
        q_emb = self.embed_fn(query)
        dense_scores = self._cosine_similarities(q_emb)
        now = datetime.now()
        combined = []
        for i, (entry, d_score) in enumerate(
            zip(self.entries, dense_scores)
        ):
            if recency_boost:
                age_h = (now - entry.timestamp).total_seconds() / 3600
                recency = np.exp(-self.decay_rate * age_h)
                score = (1 - self.recency_weight) * d_score \
                    + self.recency_weight * recency
            else:
                score = d_score
            combined.append((score, i))
        combined.sort(reverse=True)
        top_k = [self.entries[i] for _, i in combined[:k]]
        # Update access metadata
        for entry in top_k:
            entry.access_count += 1
            entry.last_accessed = now
        return top_k

    def _cosine_similarities(self, query_emb: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity against all stored embeddings."""
        matrix = np.stack([e.embedding for e in self.entries])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-8)
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        return matrix_norm @ q_norm

    # -- Reflect ------------------------------------------------------------
    def reflect(self, llm_fn, k: int = 10) -> list[str]:
        """
        Meta-cognitive reflection: retrieve recent memories,
        synthesize higher-level insights, and store them back.
        """
        if len(self.entries) < 3:
            return []
        # Retrieve recent high-importance memories
        recent = sorted(
            self.entries, key=lambda e: e.timestamp, reverse=True
        )[:k]
        context = "\n".join(f"- {e.content}" for e in recent)
        # Ask LLM to generate insights
        prompt = (
            "Given these recent memories, extract 2-3 high-level "
            "insights or patterns:\n" + context
        )
        raw_insights = llm_fn(prompt)
        # Store each insight as a high-importance memory
        insights = []
        for line in raw_insights.strip().split("\n"):
            line = line.strip().lstrip("-*").strip()
            if len(line) > 20:
                self.write(
                    f"[INSIGHT] {line}",
                    importance=0.9,
                    check_duplicates=True,
                )
                insights.append(line)
        return insights

    def get_stats(self) -> dict:
        """Return memory statistics for monitoring."""
        return {
            "total_entries": len(self.entries),
            "avg_importance": float(
                np.mean([e.importance for e in self.entries])
            ) if self.entries else 0.0,
            "oldest_entry": min(
                (e.timestamp for e in self.entries), default=None
            ),
        }
```

**代码清单 17.1：** 带嵌入、重要性评分与混合检索的向量存储记忆。

### 17.10.2 分层记忆管理器

受 MemGPT [316] 启发，此模式将记忆组织为三个层级：**热层**（hot，上下文内、即时访问）、**温层**（warm，向量存储、快速检索）与**冷层**（cold，归档、容量无限）。条目依据访问频率与重要性自动升降级——类比 CPU 的缓存层级。

```python
from enum import Enum
from collections import OrderedDict

class MemoryTier(Enum):
    HOT = "hot"    # In-context: immediate access
    WARM = "warm"  # Vector store: fast retrieval
    COLD = "cold"  # Archival: slow but unlimited

class HierarchicalMemoryManager:
    """
    Three-tier memory manager inspired by MemGPT.
    Hot tier is an LRU cache; warm is a vector store;
    cold is append-only archival storage.
    """
    def __init__(
        self,
        vector_store: VectorMemoryStore,
        hot_capacity: int = 20,          # max entries in hot tier
        warm_capacity: int = 5_000,
        llm_summarize_fn=None,           # callable for summarization
    ):
        self.vector_store = vector_store
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self.summarize = llm_summarize_fn
        # Hot tier: ordered dict for LRU semantics
        self.hot: OrderedDict[str, MemoryEntry] = OrderedDict()
        # Cold tier: append-only list (would be a DB in production)
        self.cold: list[MemoryEntry] = []

    # -- Page-in: promote warm -> hot ---------------------------------------
    def page_in(self, query: str, k: int = 3) -> list[MemoryEntry]:
        """
        Retrieve from warm store and promote to hot tier.
        Evicts least-recently-used hot entries if needed.
        """
        candidates = self.vector_store.retrieve(query, k=k)
        promoted = []
        for entry in candidates:
            key = entry.content[:64]   # use prefix as key
            if key not in self.hot:
                if len(self.hot) >= self.hot_capacity:
                    self._evict_hot()
                self.hot[key] = entry
                self.hot.move_to_end(key)
                promoted.append(entry)
        return promoted

    def _evict_hot(self):
        """Evict LRU entry from hot tier back to warm."""
        # OrderedDict: first item is LRU
        key, entry = self.hot.popitem(last=False)
        # Re-insert into warm store (already there, just update access)
        # In a real system, we'd update the warm store's metadata

    # -- Write with tier assignment -----------------------------------------
    def write(
        self,
        content: str,
        importance: float = 0.5,
        tier: MemoryTier = MemoryTier.WARM,
    ) -> MemoryEntry:
        """Write to the appropriate tier."""
        if tier == MemoryTier.HOT:
            entry = MemoryEntry(
                content=content,
                embedding=self.vector_store.embed_fn(content),
                importance=importance,
            )
            key = content[:64]
            if len(self.hot) >= self.hot_capacity:
                self._evict_hot()
            self.hot[key] = entry
            return entry
        elif tier == MemoryTier.WARM:
            return self.vector_store.write(content, importance=importance)
        else:  # COLD
            entry = MemoryEntry(
                content=content,
                embedding=np.array([]),    # no embedding for cold
                importance=importance,
            )
            self.cold.append(entry)
            return entry

    # -- Summarize and compress --------------------------------------------
    def compress_hot_to_warm(self) -> Optional[str]:
        """
        Summarize hot tier contents and write summary to warm.
        Called when hot tier is full and new important content arrives.
        """
        if not self.hot or not self.summarize:
            return None
        hot_contents = "\n".join(
            f"- {e.content}" for e in self.hot.values()
        )
        summary = self.summarize(
            f"Summarize these memory entries concisely:\n{hot_contents}"
        )
        self.vector_store.write(summary, importance=0.7)
        return summary

    # -- Unified retrieval --------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """
        Retrieve from all tiers, prioritizing hot.
        Returns up to k entries sorted by relevance.
        """
        results = []
        # 1. Check hot tier (exact + semantic)
        q_emb = self.vector_store.embed_fn(query)
        for entry in self.hot.values():
            if entry.embedding.size > 0:
                sim = float(
                    np.dot(q_emb, entry.embedding)
                    / (np.linalg.norm(q_emb) * np.linalg.norm(entry.embedding) + 1e-8)
                )
                if sim > 0.7:
                    results.append((sim + 1.0, entry))   # +1 hot bonus
        # 2. Retrieve from warm store
        warm_results = self.vector_store.retrieve(query, k=k)
        for entry in warm_results:
            results.append((0.5, entry))
        # 3. Deduplicate and sort
        seen = set()
        final = []
        for score, entry in sorted(results, reverse=True):
            key = entry.content[:64]
            if key not in seen:
                seen.add(key)
                final.append(entry)
                if len(final) >= k:
                    break
        return final

    def get_hot_context(self) -> str:
        """Return hot tier as a formatted context string."""
        if not self.hot:
            return ""
        lines = ["[Memory Context]"]
        for entry in list(self.hot.values())[-10:]:   # last 10
            lines.append(f"* {entry.content}")
        return "\n".join(lines)
```

**代码清单 17.2：** 实现热/温/冷三层、带自动升降级的分层记忆管理器。

### 17.10.3 记忆增强的智能体循环

此模式由 MemGPT [316] 引入并在 CoALA 框架 [329] 中形式化，通过一个「读-行-反思-写」（read–act–reflect–write）循环将记忆系统接入智能体的推理循环：在响应之前检索相关记忆，在响应之后决定存储什么。LLM 输出中的特殊 token 触发记忆操作，赋予模型对其自身持久化的自驱控制。

```python
import re
from typing import Any

class MemoryAugmentedAgent:
    """
    An LLM agent with a full read-act-reflect-write memory cycle.
    Implements the MemGPT-style self-directed memory management.
    """
    SYSTEM_PROMPT = """You are a memory-augmented AI assistant.
You have access to persistent memory across conversations.
At each turn you may issue memory commands:
  [MEMORY_SEARCH: <query>]   - retrieve relevant memories
  [MEMORY_WRITE: <content>]  - store important information
  [MEMORY_REFLECT]           - synthesize insights from memory
Always think step by step. Use memory to avoid repeating mistakes
and to personalize your responses."""

    def __init__(
        self,
        llm_fn,                 # callable: messages -> str
        memory_manager: HierarchicalMemoryManager,
        importance_threshold: float = 0.6,
        max_memory_tokens: int = 1500,
    ):
        self.llm = llm_fn
        self.memory = memory_manager
        self.importance_threshold = importance_threshold
        self.max_memory_tokens = max_memory_tokens
        self.conversation_history: list[dict] = []

    # -- Main agent step ----------------------------------------------------
    def step(self, user_message: str) -> str:
        """
        Full agent step:
        1. Retrieve relevant memories
        2. Construct augmented prompt
        3. Generate response (possibly with memory commands)
        4. Execute memory commands
        5. Reflect and consolidate
        6. Return response to user
        """
        # Step 1: Retrieve relevant memories
        memories = self.memory.retrieve(user_message, k=5)
        memory_context = self._format_memories(memories)
        # Step 2: Construct augmented prompt
        messages = self._build_messages(user_message, memory_context)
        # Step 3: Generate response
        raw_response = self.llm(messages)
        # Step 4: Execute any memory commands in the response
        clean_response, memory_ops = self._parse_memory_commands(
            raw_response
        )
        self._execute_memory_ops(memory_ops, user_message, clean_response)
        # Step 5: Auto-write important information
        self._auto_write(user_message, clean_response)
        # Step 6: Update conversation history
        self.conversation_history.append(
            {"role": "user", "content": user_message}
        )
        self.conversation_history.append(
            {"role": "assistant", "content": clean_response}
        )
        return clean_response

    # -- Memory retrieval and formatting -----------------------------------
    def _format_memories(self, memories: list[MemoryEntry]) -> str:
        if not memories:
            return ""
        lines = ["Relevant memories:"]
        for i, m in enumerate(memories, 1):
            age = (datetime.now() - m.timestamp).days
            lines.append(
                f"[{i}] (importance={m.importance:.1f}, "
                f"{age}d ago) {m.content}"
            )
        return "\n".join(lines)

    def _build_messages(
        self, user_message: str, memory_context: str
    ) -> list[dict]:
        system = self.SYSTEM_PROMPT
        if memory_context:
            system += f"\n\n{memory_context}"
        system += f"\n\n{self.memory.get_hot_context()}"
        messages = [{"role": "system", "content": system}]
        # Include recent conversation history (last 6 turns)
        messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": user_message})
        return messages

    # -- Memory command parsing --------------------------------------------
    def _parse_memory_commands(
        self, response: str
    ) -> tuple[str, list[dict]]:
        """Extract and remove memory commands from response."""
        ops = []
        patterns = {
            "search": r"\[MEMORY_SEARCH:\s*(.+?)\]",
            "write": r"\[MEMORY_WRITE:\s*(.+?)\]",
            "reflect": r"\[MEMORY_REFLECT\]",
        }
        clean = response
        for op_type, pattern in patterns.items():
            for match in re.finditer(pattern, response, re.DOTALL):
                content = match.group(1) if op_type != "reflect" else None
                ops.append({"type": op_type, "content": content})
                clean = clean.replace(match.group(0), "").strip()
        return clean, ops

    def _execute_memory_ops(
        self,
        ops: list[dict],
        user_msg: str,
        response: str,
    ):
        """Execute memory commands issued by the LLM."""
        for op in ops:
            if op["type"] == "search":
                results = self.memory.retrieve(op["content"], k=3)
                # Page results into hot tier for immediate use
                self.memory.page_in(op["content"], k=3)
            elif op["type"] == "write":
                self.memory.write(
                    op["content"],
                    importance=0.8,    # explicitly written = important
                    tier=MemoryTier.WARM,
                )
            elif op["type"] == "reflect":
                self._reflect()

    # -- Auto-write heuristic ----------------------------------------------
    def _auto_write(self, user_msg: str, response: str):
        """
        Automatically store important information without explicit command.
        Uses a simple heuristic: write if response contains facts,
        decisions, or user preferences.
        """
        importance_keywords = [
            "remember", "important", "note that", "you prefer",
            "your name is", "decided to", "the answer is",
            "key insight", "learned that",
        ]
        combined = (user_msg + " " + response).lower()
        if any(kw in combined for kw in importance_keywords):
            summary = f"User: {user_msg[:100]} | Agent: {response[:200]}"
            self.memory.write(
                summary,
                importance=self.importance_threshold,
                tier=MemoryTier.WARM,
            )

    # -- Reflection ---------------------------------------------------------
    def _reflect(self):
        """
        Meta-cognitive reflection: synthesize insights from recent memory.
        Stores high-level insights back into semantic memory.
        """
        recent = self.memory.retrieve("recent important events", k=10)
        if len(recent) < 3:
            return  # Not enough to reflect on
        recent_text = "\n".join(f"- {m.content}" for m in recent)
        insight_prompt = [
            {"role": "system", "content": "You extract high-level insights."},
            {"role": "user", "content":
                f"Based on these memories, what are 2-3 key insights?\n"
                f"{recent_text}\nRespond with bullet points only."},
        ]
        insights = self.llm(insight_prompt)
        # Store each insight as a high-importance semantic memory
        for line in insights.split("\n"):
            line = line.strip().lstrip("*-").strip()
            if len(line) > 20:
                self.memory.write(
                    f"[INSIGHT] {line}",
                    importance=0.9,
                    tier=MemoryTier.WARM,
                )
```

**代码清单 17.3：** 带「读-行-反思-写」循环的完整记忆增强智能体循环。

**读-行-反思-写循环**

记忆增强的智能体循环实现了一个四阶段的认知循环：

1. **读（Read）：** 行动前，检索相关记忆以指导响应。
2. **行（Act）：** 在检索到的上下文条件下生成响应。
3. **反思（Reflect）：** 周期性地从累积的记忆中合成更高层洞见。
4. **写（Write）：** 选择性地将重要的新信息提交到持久存储。

这一循环映射了军事战略中的「观察-判断-决策-行动」（OODA）循环，以及认知心理学中的「编码-存储-检索」模型。关键洞见在于：记忆不是被动的存储，而是认知的主动参与者。

## 17.11 智能体记忆的最新进展

上述记忆系统确立了基础范式。近期若干工作进一步推进了边界：

### 17.11.1 CoALA：语言智能体的认知架构

Sumers 等人 [329] 提出了**语言智能体的认知架构**（Cognitive Architectures for Language Agents, CoALA），这是一个统一框架，运用认知科学与符号 AI 的原则来组织日益庞杂的 LLM 智能体。CoALA 将一个语言智能体分解为：

- **模块化记忆：** 工作记忆（上下文窗口）、情景记忆（过往经验）、语义记忆（世界知识）与过程记忆（动作模式）——呼应我们在 17.2 节的分类。
- **结构化动作空间：** 内部动作（推理、检索、记忆写入）与外部动作（工具使用、环境交互）。
- **决策循环：** 一个广义的「感知-规划-行动」（sense–plan–act）循环，带有显式的检索与写入步骤。

CoALA 的贡献与其说是一个新系统，不如说是一种设计语言：它提供了一种系统化方式来分析现有智能体并识别缺失能力，使其成为从业者有用的参考架构。

### 17.11.2 Mem0：生产规模的记忆层

Mem0 [318] 致力于弥合研究记忆系统与生产部署之间的鸿沟。核心思想：

- **自动抽取：** Mem0 不依赖 LLM 显式发出记忆写入命令，而是自动从对话轮次中抽取显著事实，并将其固化到持久存储中。
- **基于图的记忆：** 在扁平向量存储之外，Mem0 在抽取出的实体与事实上维护一个关系图，支持多跳记忆查询（「在项目 Y 的语境下，用户对主题 X 说了什么？」）。
- **记忆压缩：** 冗余或被取代的事实会被自动合并，保持记忆存储紧凑且最新。

在 LOCOMO 基准上，Mem0 相对 OpenAI 的基线记忆取得了 26% 的相对提升，p95 延迟降低 91%，相对全上下文方法 token 成本降低 >90%。

### 17.11.3 睡眠时计算：离线记忆处理

Lin 等人 [330] 提出了**睡眠时计算**（sleep-time compute），一种智能体在用户交互之间（而非仅在查询时）处理并固化记忆的范式。其类比对象是生物睡眠——大脑在睡眠期间固化记忆并预计算有用的关联。

**工作原理。** 在空闲时段（「睡眠」）期间，智能体：

1. 在给定当前上下文的情况下，预测可能的未来查询。
2. 预计算推理链、摘要与结构化表示。
3. 存储这些预计算产物，以便测试时推理可以检索并复用它们。

**结果。** 在推理基准上，睡眠时计算将达成同等准确率所需的测试时计算量降低约 5 倍。当对同一上下文的多个相关查询进行摊销后，平均每查询成本下降 2.5 倍。当用户查询可预测时（即上下文强烈约束了会被问及的问题），该方法最为有效。

> **将记忆固化视为离线 RL**
>
> 睡眠时计算可被视为离线策略改进：在空闲时间，智能体利用其已收集的数据（过往交互）来改进记忆表示（策略），而无需新的环境交互。这与离线 RL 方法（见第 8 章）相联系——后者让智能体从一个静态的轨迹数据集中学习。

### 17.11.4 A-MEM：受卡片盒启发的智能体记忆

A-MEM [317] 引入了一种借鉴**卡片盒笔记法**（Zettelkasten method）的记忆系统——这是一种基于密集互联的原子笔记的笔记方法——为 LLM 智能体实现动态、自组织的记忆。

**关键设计原则。**

- **结构化笔记。** 每个记忆条目不是一个原始文本块，而是一个具有多种结构化属性的笔记：上下文描述、关键词、标签，以及到相关笔记的显式链接。这些元数据支持比单纯嵌入相似度更丰富的检索。
- **动态链接。** 当加入一条新记忆时，系统分析现有记忆以识别语义上有意义的连接，并建立双向链接。结果是一张知识网络，而非一份扁平列表。
- **记忆演化。** 关键在于，加入一条新笔记可能触发对现有笔记的更新——随着智能体理解的深入，优化其上下文表示与属性。这使记忆成为一种随时间改进的「活的」结构，而非静态归档。
- **智能体驱动的组织。** 不同于固定模式的记忆系统，A-MEM 让 LLM 自行决定如何组织、链接与更新记忆——使组织结构能够适应任务领域。

**结果。** 在六种基础模型上的多会话推理任务中，A-MEM 持续优于扁平向量存储、基于摘要的记忆以及图数据库方法，证明「记忆如何组织」与「存储什么」同等重要。

## 17.12 小结

智能体记忆系统是有能力 AI 智能体的基础组件，旨在应对有限上下文窗口这一根本局限。我们考察了：

- **四分类法**（工作、情景、语义、过程），它映射认知科学并反映出不同的工程要求。
- **五大架构家族：** 基于 RAG 的、基于摘要的、基于图的、键值网络，以及分层虚拟上下文（MemGPT）。
- **四项核心操作：** 写入（带重要性评分与矛盾检测）、读取/检索（带时间衰减与查询扩展）、更新（带冲突解决与固化），以及反思（元认知洞见生成）。
- **多轮与多智能体扩展：** 用户建模、会话连续性、共享记忆池与黑板架构。
- **记忆系统的 RL 训练：** 记忆操作的奖励信号、学习该记住什么、以及记忆增强的策略优化。

该领域正快速演进。关键的开放挑战包括：（1）**记忆接地（memory grounding）**——确保被检索到的记忆被忠实地纳入，而非被忽视或在之上产生幻觉；（2）**可扩展的一致性**——在大型多智能体系统中维持连贯的共享记忆；以及（3）**隐私保护的记忆**——在不损害用户数据的前提下实现个性化。随着上下文窗口增长，上下文内记忆与外部记忆之间的边界将会移动，但对选择性、结构化、可检索信息存储的根本需求将始终存在。
