# 第 28 章 速查参考

本章汇总关键公式、架构规格、API 参考与故障模式诊断,便于在开发与调试过程中快速查阅。

## 28.1 核心强化学习与对齐公式

**PPO Clip(PPO 截断):**

$$
L = \mathbb{E}\left[\min\left(r_t \hat{A}_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right], \quad r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_\text{old}(a_t \mid s_t)} \quad (28.1)
$$

**DPO(直接偏好优化):**

$$
L = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right] \quad (28.2)
$$

**GRPO:**

$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}, \quad \text{随后进行 PPO 截断更新(无 critic)} \quad (28.3)
$$

**KTO:**

$$
L = \lambda_w(1 - v(y_w)) + \lambda_l \cdot v(y_l), \quad v = \sigma\left(\beta \log\frac{\pi_\theta}{\pi_\text{ref}} - z\right) \quad (28.4)
$$

**IPO:**

$$
L = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)} - \frac{1}{2\beta}\right)^2\right] \quad (28.5)
$$

**ORPO:**

$$
L = L_\text{SFT}(y_w) - \lambda \log \sigma\left(\log \frac{\text{odds}(y_w)}{\text{odds}(y_l)}\right) \quad (28.6)
$$

**GAE(广义优势估计):**

$$
\hat{A}_t = \sum_{l=0}^{T-t}(\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad (28.7)
$$

**KL 惩罚:**

$$
R_\text{total} = r_\phi(x, y) - \beta D_\mathrm{KL}\left[\pi_\theta(y \mid x) \| \pi_\text{ref}(y \mid x)\right] \quad (28.8)
$$

**奖励模型(RM,Bradley-Terry):**

$$
L = -\mathbb{E}\left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right] \quad (28.9)
$$

**Best-of-N(N 中选优):**

$$
y^* = \arg\max_{y_i \sim \pi_\theta(\cdot \mid x),\ i=1..N} r_\phi(x, y_i) \quad (28.10)
$$

## 28.2 Transformer 与架构公式

**自注意力(Self-Attention):**

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) \cdot V \quad (28.11)
$$

**多头注意力(Multi-Head):**

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\text{head}_1, \dots, \text{head}_h) W^O, \quad \text{head}_i = \mathrm{Attn}(XW_i^Q, XW_i^K, XW_i^V) \quad (28.12)
$$

**RoPE(旋转位置编码):**

$$
f(x_m, m) = x_m e^{im\theta_j}, \quad \theta_j = 10000^{-2j/d} \quad (28.13)
$$

**LoRA:**

$$
W' = W_0 + \frac{\alpha}{r} \cdot BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k} \quad (28.14)
$$

**知识蒸馏(KD,软目标):**

$$
L_\mathrm{KD} = (1-\alpha) L_\mathrm{CE}(y, \hat{y}) + \alpha T^2 \cdot \mathrm{KL}\left(p_T^\text{teacher} \| p_T^\text{student}\right) \quad (28.15)
$$

**前馈网络(FFN,SwiGLU):**

$$
\mathrm{FFN}(x) = (\mathrm{Swish}(xW_1) \odot xW_3)W_2 \quad (28.16)
$$

## 28.3 解码方法

| 方法 | 公式 / 规则 | 关键参数 |
|---|---|---|
| 贪心解码(Greedy) | $y_t = \arg\max_v P(v \mid y_{<t})$ | — |
| 束搜索(Beam search) | 按联合概率保留前 B 个部分序列 | $B = 4$–$8$ |
| 温度(Temperature) | $P'(v) = \mathrm{softmax}(\text{logit}_v / T)$ | $T \in [0.1, 1.5]$ |
| Top-k | 除前 k 个 logit 外置零,再重新归一化 | $k = 40$–$100$ |
| Top-p(核采样,nucleus) | 保留最小的集合 $V'$ 使得 $\sum_{v \in V'} P(v) \geq p$ | $p = 0.9$–$0.95$ |
| Min-p | 保留满足 $P(v) \geq p_\text{min} \cdot P(v_\text{max})$ 的词元 | $p_\text{min} = 0.05$–$0.1$ |
| 重复惩罚(Repetition penalty) | 若 $v$ 之前出现过,$\text{logit}_v \leftarrow \text{logit}_v / \theta$ | $\theta = 1.1$–$1.3$ |

## 28.4 系统与并行

| 项目 | 公式 | 量值(70B,BF16) | 说明 |
|---|---|---|---|
| 模型显存(Model memory) | $2P$ bytes | 140 GB(仅权重) | — |
| Adam 优化器(Adam optimizer) | $2P \times 4$ bytes($m + v$) | 280 GB | — |
| 完整训练占用(Full training footprint) | $\sim 8P$ bytes | 560 GB(权重 + 优化器 + 梯度) | — |
| 每张 GPU 的 FSDP 显存 | $8P / N_\text{GPUs}$ | 8 张 GPU 时 70 GB | — |
| 生成算术强度(Gen arithmetic intensity) | $2P / 2P = 1$ FLOP/byte | 严重受限于显存 | — |
| 词元速率(Token rate,生成) | $\text{HBM\_BW} / (2P)$ | $\sim 14$ tok/s(A100,batch=1) | — |
| 每层 TP AllReduce | $2 \times 2 \cdot T^{-1} \cdot T \cdot bsd$ bytes | $\sim 188$ MB(70B,TP=8) | — |
| PP 气泡占比(PP bubble fraction) | $(P-1)/(P+M-1)$ | $P$=阶段数,$M$=微批次 | — |
| 模型算力利用率(MFU) | $\text{observed\_toks} \times 6P / \text{peak\_FLOPS}$ | 目标:$> 40\%$ | — |

## 28.5 GPU 硬件规格

| GPU | 显存 | 带宽(HBM) | BF16 TFLOPS | NVLink | 备注 |
|---|---|---|---|---|---|
| A100-80GB | 80 GB HBM2e | 2.0 TB/s | 312 | 600 GB/s | 主力卡,货源广泛 |
| H100-80GB | 80 GB HBM3 | 3.35 TB/s | 989 | 900 GB/s | 当前代,支持 FP8 |
| H200-141GB | 141 GB HBM3e | 4.8 TB/s | 989 | 900 GB/s | 大上下文 / 更少 GPU |
| B200 | 192 GB HBM3e | 8.0 TB/s | 2250 | 1800 GB/s | 下一代(2025) |

## 28.6 超参数范围

| 参数 | 典型范围 | 默认值 | 备注 |
|---|---|---|---|
| $\beta$(DPO/KTO) | 0.05–0.5 | 0.1 | 越大越保守 |
| $\epsilon$(PPO 截断) | 0.1–0.3 | 0.2 | 越大更新越激进 |
| $\gamma$(GAE 折扣) | 0.99–1.0 | 1.0 | 回合式任务用 1.0 |
| $\lambda$(GAE) | 0.9–0.99 | 0.95 | 越小偏差越大、方差越小 |
| KL 系数($\beta_\mathrm{KL}$) | 0.01–0.2 | 0.05 | 自适应至目标 KL $\approx$ 5–8 |
| LR(RLHF) | 1e-7 – 5e-6 | 5e-7 | 远低于预训练 |
| LR(SFT) | 1e-5 – 5e-5 | 2e-5 | 标准微调范围 |
| LoRA 秩 $r$ | 8–128 | 16–64 | $r$ 越大容量越大、显存越多 |
| LoRA alpha $\alpha$ | $r$ – $2r$ | $2r$ | 缩放因子;$\alpha/r$ 为有效缩放 |
| 温度(生成) | 0.6–1.0 | 0.7 | 越小候选越不多样 |
| 生成数 $K$ | 4–64 | 4–16 | 用于 GRPO / Online DPO / Best-of-N |
| 梯度裁剪范数(Grad clip norm) | 0.5–2.0 | 1.0 | 防止梯度爆炸 |

## 28.7 TRL API 速查

| 训练器(Trainer) | 方法 | 关键配置 | 数据格式 |
|---|---|---|---|
| SFTTrainer | 监督微调(Supervised FT) | `packing`,`max_seq_length` | prompt + completion |
| RewardTrainer | 奖励模型(Reward model) | `center_rewards_coefficient` | prompt + chosen + rejected |
| PPOTrainer | PPO | `init_kl_coef`,`target_kl`,`cliprange` | prompts(在线生成) |
| DPOTrainer | DPO/IPO | `beta`,`loss_type="sigmoid"/"ipo"` | prompt + chosen + rejected |
| GRPOTrainer | GRPO | `num_generations`,`beta`,`use_vllm` | prompts + reward_fn |
| OnlineDPOTrainer | Online DPO | `num_generations`,`reward_model_path` | prompts(在线生成) |
| KTOTrainer | KTO | `desirable_weight`,`undesirable_weight` | prompt + completion + label |
| ORPOTrainer | ORPO | `beta` | prompt + chosen + rejected |
| Best-of-N(手动) | Best-of-N | `n_samples` | prompts(推理) |

## 28.8 RAG 流水线公式

**余弦相似度(Cosine similarity):**

$$
\mathrm{sim}(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|} \quad (28.17)
$$

**检索(Retrieval):**

$$
D_k = \text{top-}k_{d \in C}\ \mathrm{sim}\left(\text{embed}(q), \text{embed}(d)\right) \quad (28.18)
$$

**RAG 生成(RAG generation):**

$$
P(y \mid q) = P_\mathrm{LLM}(y \mid q, D_k) \quad (28.19)
$$

**分块重叠(Chunking overlap):**

$$
\text{stride} = \text{chunk\_size} - \text{overlap} \quad (28.20)
$$

**重排序器(Reranker,cross-encoder):**

$$
\mathrm{score}(q, d) = \mathrm{MLP}\left(\mathrm{BERT}([q; d])\right) \quad (28.21)
$$

## 28.9 智能体设计模式

| 模式 | 结构 | 最适合 |
|---|---|---|
| ReAct | 思考 $\to$ 行动 $\to$ 观察 $\to$ 循环 | 通用工具使用智能体 |
| 计划-执行(Plan-and-Execute) | 计划 $\to$ 执行各步 $\to$ 修订 | 长时程、结构化任务 |
| 主管(Supervisor) | 路由 $\to$ 专职智能体 | 多领域、子任务边界清晰 |
| 蜂群(Swarm,交接 handoffs) | 智能体移交控制权 + 上下文 | 客服、升级流转 |
| 分层(Hierarchical) | 委托型智能体树 | 复杂分解 |
| 人在回路(Human-in-the-loop) | 智能体 $\to$ 审批闸门 $\to$ 继续 | 高风险、不可逆操作 |

## 28.10 智能体通信协议

| 协议 | 范围 | 传输方式 | 核心概念 |
|---|---|---|---|
| MCP | 工具集成 | stdio / HTTP+SSE | 服务端暴露工具,客户端发现并调用 |
| A2A | 智能体之间 | HTTP + JSON-RPC | 带生命周期的任务(submitted $\to$ working $\to$ done) |
| OpenAI Function Calling | 工具使用 | API 载荷 | `tools[]` 数组中的 JSON schema |

## 28.11 上下文窗口预算

$$
C \geq \underbrace{S}_\text{system} + \underbrace{M}_\text{memory/RAG} + \underbrace{T}_\text{tool defs} + \underbrace{H}_\text{history} + \underbrace{R}_\text{reserved output} \quad (28.22)
$$

128K 上下文的经验法则:

- 系统提示(System prompt):1–4K tokens(固定)
- 工具定义(Tool definitions):2–8K(随工具数量增长)
- RAG 上下文:4–16K(top-k 分块)
- 历史(History):无界增长 $\to$ 摘要 / 截断
- 预留输出(Reserved output):2–8K

## 28.12 常见故障模式与修复

| 症状 | 可能原因 | 修复 |
|---|---|---|
| 奖励上升,质量下降 | 奖励黑客(Reward hacking) | RM 集成、长度惩罚、增大 $\beta$ |
| KL 爆炸($>15$) | LR 过高或模式塌缩(mode collapse) | 降低 LR、回滚检查点 |
| 熵塌缩(Entropy collapse) | 过早收敛 | 增大熵系数、提高温度 |
| 训练 loss 为 NaN | 梯度爆炸 | 降低 LR、增大梯度裁剪、检查数据 |
| 5K 步后无改进 | 提示分布不佳 | 金发姑娘过滤(20–80% 通过率) |
| 基准回退(Benchmark regression) | 对齐税(Alignment tax) | 减少 RL 预算、用 LoRA、掺入 SFT 数据 |
| 长度单调递增 | RM 中的长度利用 | 长度惩罚、用长度控制重训 RM |
| 生成时 OOM | KV 缓存溢出 | 减小 batch、增大 TP、PagedAttention |
| 智能体无限循环 | 无最大迭代保护 | 设置 max_iterations、加循环检测 |
| 工具调用解析失败 | 输出格式不一致 | 少样本示例、受限解码 |
| RAG 返回无关文档 | 嵌入 / 分块不佳 | 重排序器、混合检索、更小分块 |
| 多智能体死锁 | 循环依赖 | DAG 强制、每智能体超时 |

## 28.13 方法选择决策树

1. 是否有成对偏好(chosen + rejected)?
   - 标签有噪声 $\to$ IPO
   - 显存受限且尚未做 SFT $\to$ ORPO
   - 数据干净、算力有限 $\to$ DPO
   - DPO 遇到瓶颈,想要探索 $\to$ Online DPO
2. 仅有二值反馈(点赞 / 踩)? $\to$ KTO
3. 有可验证奖励(数学 / 代码)? $\to$ GRPO
4. 需要最高质量、不计成本? $\to$ PPO
5. 想要免训练的改进? $\to$ Best-of-N

## 28.14 评估指标

| 指标 | 取值范围 | 衡量内容 |
|---|---|---|
| 困惑度(Perplexity) | $[1, \infty)$ | 模型的意外程度;越低 = 语言建模越好 |
| 胜率(Win Rate,相对基线) | $[0, 1]$ | 由评判者 / 人类偏好的输出占比 |
| BLEU | $[0, 1]$ | 与参考的 n-gram 重叠(侧重精确率) |
| ROUGE-L | $[0, 1]$ | 与参考的最长公共子序列 |
| Pass@k | $[0, 1]$ | k 个代码样本中至少 1 个通过测试的概率 |
| MMLU / GPQA | $[0, 1]$ | 知识 / 推理基准的多选准确率 |
| HumanEval | $[0, 1]$ | 生成代码的功能正确性 |
| 忠实度(Faithfulness,RAG) | $[0, 1]$ | 由检索上下文支持的主张占比 |
| 上下文相关性(Context Relevancy) | $[0, 1]$ | 与查询相关的检索内容占比 |
| 答案相关性(Answer Relevancy) | $[0, 1]$ | 答案切合问题的程度 |

## 28.15 推理与测试时扩展

| 方法 | 计算开销 | 机制 |
|---|---|---|
| 思维链(Chain-of-Thought, CoT) | 1.5–3× tokens | 在提示中要求"一步步思考" |
| 自洽性(Self-Consistency) | N× 生成 | 采样 N 条 CoT 路径,对最终答案多数投票 |
| 思维树(Tree-of-Thought, ToT) | $B \times D$× 生成 | 在推理树上做 BFS/DFS;评估各分支 |
| Best-of-N | N× 生成 | 采样 N 个,用 RM 打分,选最高者 |
| 束搜索(Beam search,用于推理) | B× 生成 | 维持前 B 条部分推理链 |
| 预算强制(Budget forcing) | 可变 | 动态为更难的问题分配更多 token |
| 验证(Verification,ORM/PRM) | N× 生成 + 打分 | 生成 N 个解,按结果 / 过程 RM 排序 |

## 28.16 记忆系统类型

| 类型 | 存储 | 用途 |
|---|---|---|
| 工作记忆(Working memory) | 上下文窗口 | 当前对话、即时工具结果 |
| 情景记忆(Episodic memory) | 向量存储 | 过往交互、用户偏好、会话历史 |
| 语义记忆(Semantic memory) | 知识图谱 / 嵌入 | 事实、概念、领域知识 |
| 过程记忆(Procedural memory) | 技能库 / 代码 | 操作流程、已学工作流 |

## 28.17 MCP 速查

| 原语(Primitive) | 方向 | 有副作用? | 用途 |
|---|---|---|---|
| 工具(Tools) | 客户端 $\to$ 服务端 | 是 | 执行动作(创建、修改、删除) |
| 资源(Resources) | 客户端 $\to$ 服务端 | 否(只读) | 读取数据(文件、DB 记录、配置) |
| 提示(Prompts) | 客户端 $\to$ 服务端 | 否 | 常用任务的可复用模板 |
| 采样(Sampling) | 服务端 $\to$ 客户端 | 否 | 服务端请求客户端进行 LLM 生成 |

- 传输(Transport):stdio(本地子进程)或 HTTP+SSE(远程、可流式)。
- 发现(Discovery):客户端在连接初始化时调用 `tools/list`、`resources/list`、`prompts/list`。
- 工具注解(Tool annotations):`readOnlyHint`、`destructiveHint`、`idempotentHint`、`openWorldHint`。

## 28.18 A2A 协议速查

| 概念 | 说明 |
|---|---|
| 智能体卡片(Agent Card) | 位于 `/.well-known/agent.json` 的 JSON —— 名称、技能、支持的内容类型 |
| 任务(Task) | 工作单元:id、状态、产物。生命周期:submitted $\to$ working $\to$ completed/failed |
| 消息(Message) | 任务内的通信单元(role:user/agent,parts:text/file/data) |
| 产物(Artifact) | 智能体产生的输出(结构化数据、文件、生成内容) |
| 推送通知(Push Notifications) | 针对长运行任务的 webhook 式更新(经 `tasks/pushNotification/set`) |

关键端点:`tasks/send`(创建 / 更新)、`tasks/get`(轮询状态)、`tasks/sendSubscribe`(SSE 流)。

## 28.19 智能体框架对比

| 框架 | 编排方式 | 多智能体 | 最适合 |
|---|---|---|---|
| LangGraph | 显式状态图 | 条件路由 | 生产:持久化、HITL、精细控制 |
| OpenAI Agents SDK | 声明式交接 | 基于交接 | 简洁:护栏、追踪、快速上手 |
| AutoGen(AG2) | 对话驱动 | GroupChat | 原型:代码执行、研究 |
| CrewAI | 基于角色的团队 | 顺序 / 并行 | 低代码:快速演示、简单流水线 |
| Google ADK | 会话 + 事件 | A2A 原生 | 企业:产物管理、多模态 |

## 28.20 智能体式强化学习公式

**轨迹 GRPO(Trajectory GRPO):**

$$
\hat{A}_i = \frac{R(\tau_i) - \mu_G}{\sigma_G}, \quad R(\tau_i) = \sum_t r(\tau_i)_t \quad (28.23)
$$

**智能体奖励(Agent reward):**

$$
R = w_1 R_\text{task} + w_2 R_\text{efficiency} + w_3 R_\text{safety}, \quad R_\text{eff} = \max\left(0, 1 - \text{steps}/N_\text{max}\right) \quad (28.24)
$$

**掩码(Masking):**

$$
L = \sum_{t \in \text{agent tokens}} \min\left(r_t \hat{A}_t,\ \mathrm{clip}(r_t)\hat{A}_t\right) \quad \text{(屏蔽环境输出)} \quad (28.25)
$$

**Pass@k:**

$$
1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}, \quad n = \text{总样本数},\ c = \text{正确数} \quad (28.26)
$$

## 28.21 智能体安全检查清单

| 威胁 | 层次 | 缓解措施 |
|---|---|---|
| 提示注入(直接) | 输入 | 输入校验、指令层级、分隔符 |
| 提示注入(间接) | 工具输出 | 将工具输出视为不可信;不遵循检索文档中的指令 |
| 工具误用 | 执行 | 最小权限;`destructiveHint` 闸门;沙箱化 |
| 数据外泄(Data exfiltration) | 输出 | 输出过滤;限制工具对允许域的访问 |
| 过度自主(Excessive autonomy) | 架构 | 最大迭代数;成本预算;人工审批闸门 |
| 混淆代理(Confused deputy) | 多智能体 | 校验任务来源;基于能力的访问控制 |

## 28.22 智能体评估指标

| 指标 | 公式 / 定义 | 目标 |
|---|---|---|
| 任务成功率(Task Success Rate, TSR) | 正确完成数 / 总任务数 | $> 85\%$(生产) |
| 完成步数(Steps to completion) | 每个成功任务的平均智能体动作数 | 越低 = 越高效 |
| 单任务成本(Cost per task) | 总 tokens × 单价 | 视预算而定 |
| 延迟(TTFC,首字符时延) | 从请求到首个有用输出的时间 | 交互式 $< 5$ s |
| 工具调用准确率(Tool call accuracy) | 正确工具选择数 / 总调用数 | $> 90\%$ |
| 恢复率(Recovery rate) | 成功重试数 / 初始失败数 | $> 60\%$ |
| 人工升级率(Human escalation rate) | 需人工的任务数 / 总任务数 | $< 15\%$ |

## 28.23 关键智能体基准

| 基准 | 领域 | 指标 | SOTA(2025) |
|---|---|---|---|
| SWE-bench Verified | 软件工程 | 已解决 issue 占比 | $\sim 70\%$ |
| WebArena | 网页浏览 | 任务成功率 | $\sim 40\%$ |
| OSWorld | 桌面计算机操作 | 任务成功率 | $\sim 25\%$ |
| GAIA | 通用 AI 助手 | 精确匹配准确率 | $\sim 75\%$(L1) |
| Tau-bench | 工具使用可靠性 | 通过率(5 次试验) | $\sim 65\%$ |
| HumanEval / MBPP | 代码生成 | Pass@1 | $> 95\%$ |
