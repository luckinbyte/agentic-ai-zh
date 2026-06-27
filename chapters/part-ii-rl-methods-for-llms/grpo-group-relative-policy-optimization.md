# 第 7 章 GRPO —— 组相对策略优化(Group Relative Policy Optimization)

组相对策略优化(Group Relative Policy Optimization, GRPO)[14] 是一种专门为语言模型设计的强化学习(RL)算法,它消除了对独立价值网络(critic)的需求。GRPO 由 DeepSeek 在 DeepSeekMath 工作中首次提出,并随后扩展到 DeepSeek-R1 [15],已迅速成为 LLM 训练中最主流的 RL 方法——被大多数开源对齐框架(TRL、OpenRLHF、veRL)采纳为默认算法。

其核心思想看似简单:与其训练一个神经网络来预测期望奖励(PPO 中的 critic),GRPO 通过为同一个提示(prompt)生成多个回复,并以该组的奖励统计量作为基线(baseline)来经验性地估计这一期望。这一做法从内存中移除了整整一个模型,使工程复杂度减半,而且令人惊讶的是,它往往优于 PPO,因为经验基线比训练不良的价值函数更准确。

GRPO 在以下场景尤为有效:

- 具有可验证奖励(verifiable reward)的推理任务(数学、代码),其中二元正确性提供了干净的信号。
- 大型模型(70B+),此时移除 critic 所节省的内存至关重要。
- 多回合(multi-turn)与智能体(agentic)场景,此时跨工具调用的价值估计难以处理。

本章涵盖 GRPO 的动机、算法、关键变体(Dr. GRPO、DAPO、2-GRPO、GDPO),以及使用 TRL 的实践实现。

## 7.1 动机

PPO 的价值模型(critic)在语言建模中存在三大问题:

1. **内存**:价值头(value head)与策略主干(policy backbone)共享(70B 模型需要 140GB)。如果使用独立的 critic,内存会翻倍。
2. **准确性**:为部分序列预测期望奖励极其困难。价值函数常常出错 → 错误的优势 → 错误的梯度方向。
3. **训练**:价值头需要大量样本才能收敛。在 RL 早期阶段,它给出嘈杂的预测,会破坏策略学习的稳定性。

GRPO 的关键洞见 [14]:与其学习 $V(s)$,不如从一组样本中经验性地估计它。为同一个提示生成 $G$ 个回复,计算其奖励,并用组统计量作为基线。

## 7.2 算法

1. 对每个提示 $x$,采样 $G$ 个补全(completions):$\{y_1, \dots, y_G\} \sim \pi_\theta(\cdot \mid x)$
2. 对每个回复打分:$r_i = R(x, y_i)$
3. 在组内归一化:$\hat{A}_i = \dfrac{r_i - \mu_G}{\sigma_G}$,其中 $\mu_G = \dfrac{1}{G}\sum_j r_j$,$\sigma_G = \text{std}(\{r_j\})$
4. 用这些优势应用 PPO 风格的截断更新:

$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}, \qquad \mathcal{L} = \mathbb{E}\left[ \min\left( r_t(\theta)\hat{A}_i,\ \text{clip}\big(r_t(\theta),\, 1 \pm \epsilon\big)\hat{A}_i \right) \right] - \beta\, D_{KL}\!\left[\pi_\theta \,\|\, \pi_{ref}\right] \quad (7.1)
$$

**组归一化为何有效**

- **组均值近似 $V(s)$**:如果对同一个提示采样足够多的回复,它们的平均奖励就是对期望奖励(= 价值函数)的一个蒙特卡洛估计。
- **高于均值 = 好动作**:$\hat{A}_i > 0$ 表示该回复比该提示下的平均水平更好,对其加强。
- **低于均值 = 坏动作**:$\hat{A}_i < 0$ 表示比平均水平更差,对其抑制。
- **归一化**:除以 $\sigma_G$ 确保优势对不同奖励范围的提示保持尺度不变(scale-invariant)。

DeepSeek-R1 的突破 [15]:使用二元正确性奖励($r = 1$ 若答案正确,否则 $r = 0$)训练的纯 GRPO,在数学/代码任务上自发地发展出了思维链(chain-of-thought)推理、自我验证(self-verification)和错误纠正能力——而无需任何显式的相关指令。

![图 7.1:GRPO 实战示意:为单个数学提示采样了 G=5 个回复。其中三个正确(r=1),两个错误(r=0)。组均值 µ_G=0.6 充当基线;正确回复获得正优势(被加强),错误回复获得负优势(被抑制)。](images/part-ii-rl-methods-for-llms/grpo-group-relative-policy-optimization/grpo-group-relative-policy-optimization-p159-01.png)

## 7.3 TRL 实现

下面展示了使用 HuggingFace TRL 的一个最小可用示例。

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,                 # G = 组大小
    temperature=1.0,                   # 高温以获得组内多样性
    max_completion_length=2048,        # 最大回复长度
    beta=0.04,                         # KL 惩罚系数
    learning_rate=1e-6,
    per_device_train_batch_size=2,     # 每设备 2 个提示(×8 生成 = 16 个回复)
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    logging_steps=10,
    # 使用 vLLM 加速生成(对 GRPO 至关重要,因为生成量是 8 倍)
    use_vllm=True,
    vllm_gpu_memory_utilization=0.7,
)


# 奖励函数:数学题的二元正确性
def reward_fn(completions, prompts, **kwargs):
    """返回浮点数列表:正确为 1.0,错误为 0.0。"""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        answer = extract_answer(completion)
        expected = get_ground_truth(prompt)
        rewards.append(1.0 if answer == expected else 0.0)
    return rewards


# 可以组合多个奖励函数!
def format_reward_fn(completions, **kwargs):
    """对使用规范的 LaTeX 排版给予奖励。"""
    return [0.5 if "\\boxed{" in c else 0.0 for c in completions]


trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=[reward_fn, format_reward_fn],   # 多目标!
    train_dataset=math_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## 7.4 组大小分析

| $G$ | 信号质量 | 计算量 | 适用场景 |
|---|---|---|---|
| 2 | 非常嘈杂(抛硬币) | 低 | 永不推荐——对稳定学习而言太嘈杂 |
| 4 | 中等 | 中等 | 快速实验、简单任务(通过率 > 50%) |
| 8 | 良好(标准) | 高 | 默认值。对多数任务的良好平衡 |
| 16 | 优秀 | 很高 | 困难任务(通过率 < 20%),需要多次尝试才能得到正样本 |
| 32 | 近乎完美 | 极高 | 仅当你拥有海量算力且任务极难时 |

**关键:组内必须同时包含成功与失败样本**

- 如果所有 $G$ 个回复都正确($r_i = 1,\ \forall i$):所有优势 = 0,没有学习信号!
- 如果全部错误:同样的问题。提示的难度必须与模型的能力匹配。
- **金发姑娘规则(Goldilocks rule)**:为当前模型筛选通过率在 20–80% 的提示。随着模型改进,每 500 步重新筛选一次。

## 7.5 GRPO 的变体与扩展

### 7.5.1 GRPO 组内的多样性

**RL 训练中的模式坍缩(mode collapse)**

在没有多样性压力的情况下,经 RL 训练的 LLM 会坍缩到一组狭窄的高奖励回复:

- 模型为每种问题类型学到一种「模板化」答案。
- 熵(entropy)迅速下降;模型变得确定性化。
- 奖励黑客(reward hacking)变得更容易(狭窄的输出更易被利用)。
- 泛化能力受损:模型记住的是奖励模式,而非推理过程。

KL 惩罚项 $\beta D_{KL}[\pi_\theta \| \pi_{ref}]$ 是主要的多样性机制,但单独使用并不足够。

**GRPO 组的多样性**

GRPO 为每个提示生成 $N$ 个回复并使用组内排序。组内的多样性至关重要:

- 高温度($\tau = 0.8$–$1.0$):确保回复有变化,以便进行有意义的比较。
- 较大的 $N$(8–16):更多样本 = 更可能同时包含好的与坏的方法。
- DAPO 的「无重复(No Repeat)」惩罚:拒绝组内的重复回复以强制探索。
- 若所有 $N$ 个回复都相同:优势为零,没有学习信号。
- 若回复过于多样(随机):奖励信号嘈杂,学习缓慢。

最佳平衡点:选择既能产生不同方法又紧扣主题的温度。

表 7.1:RL 训练中促进多样性的方法。

| 方法 | 如何促进多样性 |
|---|---|
| 熵奖励(Entropy bonus) | 在奖励中加入 $\alpha H(\pi_\theta)$,直接惩罚低熵(确定性)策略。 |
| KL 惩罚 | $-\beta D_{KL}[\pi_\theta \| \pi_{ref}]$ 防止坍缩到单一模式。 |
| 拒绝采样(Rejection sampling) | 生成大量候选,按奖励保留前 k 个。自然地选择出多样且高质量的回复。 |
| Best-of-N | 推理时:生成 $N$ 个回复,全部打分,返回最佳者。多样性来自采样。 |
| 带多样对的 DPO | 在所选/被拒方法不同(而非仅质量不同)的样本对上训练。 |
| 多奖励(Multi-reward) | 使用多个奖励模型(安全性、有用性、代码质量),防止坍缩到单一维度。 |

**多样性与质量的权衡**

更多多样性并不总是更好:

- 过多多样性(高熵)= 随机、无用的回复。
- 过少多样性(低熵)= 重复、被奖励黑客攻击的回复。
- 监控:在训练期间跟踪回复熵、唯一 n-gram 比例和奖励分布宽度。如果三者同时下降,说明出现了坍缩问题。

**用于 RL 数据收集的语言化采样(Verbalized Sampling)**

后训练对齐(RLHF、DPO)往往会因典型性偏差(typicality bias)而降低输出多样性:人类标注者系统性地偏好熟悉、「典型」的文本,而非新颖的替代方案。这种模式坍缩是一种数据层面的现象,并非纯算法性的。

语言化采样(Verbalized Sampling, VS)[111] 是一种免训练的提示策略,它通过让模型在单次生成中显式地用语言表达多个回复上的概率分布,来规避这种坍缩。

**语言化采样 —— 核心思想**

与其采样单个回复(这会坍缩到模式),不如提示模型输出多个候选回复及其概率:

> 「生成 5 个关于咖啡的笑话,以及它们对应的概率。」

模型会产生类似如下的列表:

1. 笑话 A(概率:0.35)
2. 笑话 B(概率:0.25)
3. 笑话 C(概率:0.20)
4. 笑话 D(概率:0.12)
5. 笑话 E(概率:0.08)

然后从这一语言化分布中采样。由于模型显式地表示了完整分布(而不仅是 argmax),那些概率较低但富有创意/多样性的回复变得可被获取。

```python
# 语言化采样:提示模型输出分布
def verbalized_sample(model, tokenizer, task, n=5):
    prompt = (
        f"{task}\n\n"
        f"Generate {n} different responses and assign a probability "
        f"to each (probabilities should sum to 1.0). "
        f"Format: [response] (probability: X.XX)"
    )
    output = model.generate(
        tokenizer(prompt, return_tensors="pt").input_ids,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
    )
    # 从输出中解析回复与概率
    responses, probs = parse_verbalized_distribution(
        tokenizer.decode(output[0])
    )
    # 从语言化分布中采样
    import random
    chosen = random.choices(responses, weights=probs, k=1)[0]
    return chosen
```

**语言化采样为何有效**

- **绕过模式坍缩**:从对齐模型的标准采样高度集中在一种或两种「安全」回复上。VS 迫使模型表述出它知道但通常不会浮现的替代方案。
- **多样性是语义层面的**:与温度缩放(词法噪声)不同,VS 产生真正不同的方法——模型会针对不同的选项进行推理。
- **随能力增长而增强**:更强的模型产生校准更好的语言化分布——它们从 VS 中获益更多(创意写作中有 1.6–2.1 倍的多样性提升)。
- **免训练**:无需微调或修改解码;在推理时与任何指令跟随模型兼容。
- **对 GRPO 而言**:用 VS 生成每个提示的 $G$ 个候选回复,可确保组内包含语义上多样的方法,而非表面层次的变化。

在深入扩展之前,让我们简要回顾前面几节确立的基线 GRPO 算法。其核心机制——采样一组补全、归一化其奖励、并应用截断的策略梯度——以其简洁性而优雅。然而,从业者很快发现了若干具体的失效模式:稀释梯度的预训练偏差(Dr. GRPO)、限制探索的对称截断(DAPO)、浪费的大组规模(2-GRPO)、以及多目标场景下的奖励尺度失衡(GDPO)。下面各小节依次逐一讨论。

**GRPO 基线回顾**

给定提示 $q$,从当前策略 $\pi_\theta$ 采样 $G$ 个补全 $\{o_1, \dots, o_G\}$。计算奖励 $\{r_1, \dots, r_G\}$ 并归一化:

$$
\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}, \qquad \mu_r = \frac{1}{G}\sum_{i=1}^{G} r_i, \qquad \sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i - \mu_r)^2}.
$$

截断替代损失(每 token)为:

$$
\mathcal{L}_{GRPO} = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left( \rho_{i,t}\hat{A}_i,\ \text{clip}\big(\rho_{i,t},\, 1-\epsilon,\, 1+\epsilon\big)\hat{A}_i \right),
$$

其中 $\rho_{i,t} = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \,/\, \pi_{old}(o_{i,t} \mid q, o_{i,<t})$。

### 7.5.2 DAPO —— 动态自适应策略优化(Dynamic Adaptive Policy Optimization)

**为什么需要 DAPO?**

基线 GRPO 使用对称截断:无论策略想要提高还是降低某个 token 的概率,它受到的约束是相同的。但探索(exploration)与利用(exploitation)有不同的风险特征。提高一个好 token 的概率通常是安全的;而抑制一个恰巧出现在糟糕补全中的 token,若该 token 本身是中性的,则可能是灾难性的错误。DAPO [184] 引入了五项针对性修正,共同显著提升了训练稳定性和最终性能。

**组件 1 —— 非对称截断(Clip-Higher)**

标准 PPO/GRPO 在 $[1-\epsilon,\ 1+\epsilon]$ 处对称地截断重要性比率(importance ratio)。DAPO 将其替换为非对称区间:

$$
\text{clip}_{DAPO}(\rho, A) = \begin{cases} \text{clip}(\rho,\, 1-\epsilon,\, 1+\epsilon_{high}) & \text{若 } A > 0 \\ \text{clip}(\rho,\, 1-\epsilon,\, 1+\epsilon) & \text{若 } A \le 0 \end{cases}
$$

其中 $\epsilon_{high} > \epsilon$(典型取值:$\epsilon = 0.2$,$\epsilon_{high} = 0.28$)。当优势为正时,允许策略朝好 token 移动得更远;当优势为负时,应用通常的保守截断以避免过度抑制。

**组件 2 —— Token 级损失聚合**

基线 GRPO 将损失除以序列数 $G$。DAPO 改为除以所有序列上的 token 总数:

$$
\mathcal{L}_{token} = -\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \min\left( \rho_{i,t}\hat{A}_i,\ \text{clip}_{DAPO}(\rho_{i,t}, \hat{A}_i)\hat{A}_i \right).
$$

这防止了长补全仅因其包含更多 token 就主导梯度信号。

**组件 3 —— 超长过滤(Overlong Filtering)**

当一个补全被截断(在最大长度预算内没有 EOS token)时,它提供的是误导性信号:模型会因那些生成正确、但恰好出现在截断边界之前的 token 而受罚。DAPO 将这些补全完全掩码:

$$
m_i = \mathbb{1}[EOS \in o_i], \qquad \mathcal{L}_{filtered} = -\frac{\sum_i m_i\sum_t(\cdots)}{\sum_i m_i|o_i|}.
$$

**组件 4 —— 软超长惩罚(Soft Overlong Punishment)**

而非硬掩码,一种更软的变体应用一种长度惩罚,它在补全接近最大长度 $L_{max}$ 时平滑增长:

$$
r_i \leftarrow r_i - \lambda \cdot \max\!\left(0,\ \frac{|o_i| - L_{cache}}{L_{max} - L_{cache}}\right),
$$

其中 $L_{cache}$ 是一个「安全」长度阈值。

**组件 5 —— 动态采样(Dynamic Sampling)**

DAPO 会重新采样那些整组补全获得相同奖励(全对或全错)的提示,因为这样的组在归一化后贡献的梯度为零。这使得有效批次大小在整个训练过程中保持稳定。

**DAPO 在 TRL 中**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    # 非对称截断
    epsilon=0.2,
    epsilon_high=0.28,                  # Clip-Higher
    # Token 级损失
    loss_type="dapo",                   # 启用 token 级聚合
    # 超长过滤
    mask_truncated_completions=True,
    # 生成预算
    max_completion_length=1024,
    num_generations=8,
    # 注意:DAPO 损失内部处理零方差组的过滤
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

**何时使用 DAPO**

- 长篇推理任务,补全经常触及长度上限。
- 任何在训练中观察到奖励方差坍缩的场景。
- 当基线 GRPO 显示不稳定(损失尖峰、熵坍缩)时。
- 推荐作为对多数任务基线 GRPO 的直接改进(drop-in improvement)。

### 7.5.3 GSPO —— 组序列策略优化(Group Sequence Policy Optimization)

**异策略问题**

GRPO 按 token 截断重要性比率。但一个 500 token 的序列,即使每个单独的比率都在 $[1-\epsilon,\ 1+\epsilon]$ 内,其每 token 比率的乘积也可能达到天文数字般大或小。当在同一批次上进行多步梯度更新(异策略)时,这种不匹配会迅速增长,截断边界在序列层面变得毫无意义。

GSPO [185] 将序列级重要性权重定义为每 token 比率的几何均值,等于全序列概率比率的 $|o_i|$ 次方根:

$$
s_i(\theta) = \left(\frac{\pi_\theta(o_i \mid q)}{\pi_{old}(o_i \mid q)}\right)^{1/|o_i|} = \exp\!\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{old}(o_{i,t} \mid q, o_{i,<t})}\right).
$$

这是长度归一化的序列概率比率。GSPO 损失对每序列裁剪这一个标量:

$$
\mathcal{L}_{GSPO} = -\frac{1}{G}\sum_{i=1}^{G}\min\left( s_i(\theta)\hat{A}_i,\ \text{clip}\big(s_i(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\hat{A}_i \right).
$$

**GSPO 与 GRPO 截断的对比**

- **GRPO**:独立地裁剪每个 $|o_i|$ 个每 token 比率。一个序列可能所有比率都在界内,乘积比率却达到 $10^{50}$。
- **GSPO**:每序列裁剪一次几何均值。保证序列级策略移动是有界的。
- GSPO 对异策略 IS 在理论上是正确的;GRPO 是一种近似。

**GSPO 在 TRL 中**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    # 序列级重要性采样 —— GSPO 模式
    importance_sampling_level="sequence",
    # 异策略:复用每个批次进行多步梯度更新
    steps_per_generation=4,
    num_generations=8,
    epsilon=0.2,
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
)
```

**何时使用 GSPO**

GSPO 在 `steps_per_generation > 1`(异策略训练)时最有益。对于纯同策略训练(`steps_per_generation = 1`),与 GRPO 的差异可忽略。异策略训练可以大幅降低生成成本(最昂贵的步骤),使 GSPO + 异策略成为一个强有力的效率选择。

### 7.5.4 Dr. GRPO —— 去偏奖励 GRPO(Debiased Reward GRPO)

**预训练偏差问题**

标准 GRPO 在组内归一化优势,但预训练分布引入了一种系统性偏差:在预训练数据中常见的 token 会获得大梯度,即使它们不携带任何任务相关信息。Dr. GRPO [186] 识别并校正这一偏差,把梯度信号聚焦在信息性 token 上。

Dr. GRPO 修改每 token 的梯度权重,以计入该 token 对奖励信号的边际贡献。模型已经赋予高概率(与奖励无关)的 token 会被降权:

$$
w_{i,t} = \hat{A}_i \cdot \big(1 - \pi_{ref}(o_{i,t} \mid q, o_{i,<t})\big),
$$

其中 $\pi_{ref}$ 是参考(预训练)模型。这是一种 token 效率(token efficiency)形式:梯度集中在策略真正需要改变的那些 token 上。

**Dr. GRPO 在 TRL 中**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    loss_type="dr_grpo",
    num_generations=8,
    beta=0.04,                          # KL 惩罚系数
)
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,                # token 加权所需
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
)
```

**何时使用 Dr. GRPO**

- 当训练任务在预训练与 RL 之间存在较大词表不匹配时。
- 当观察到常见填充 token 主导梯度时。
- 与接近初始策略的参考模型搭配良好。

### 7.5.5 2-GRPO —— 最小双采样 GRPO(Minimal Two-Rollout GRPO)

**「It Takes Two(成双成对)」洞见**

「It Takes Two」论文 [187] 从经验和理论上证明,在多数推理基准上,$G = 2$(每个提示仅两个补全)的 GRPO 匹配甚至超过 $G = 16$ 的 GRPO。这令人惊讶——为什么更少的样本就够了?

关键洞见在于,GRPO 的有效性主要并非来自准确的优势估计(那需要大的 $G$),而是来自一个结构上与 DPO 类似的隐式对比目标:

$$
\mathcal{L}_{\text{2-GRPO}} \approx -\mathbb{E}_{(o_+, o_-)\sim\pi_\theta}\!\left[ \log\sigma\!\left( \beta\log\frac{\pi_\theta(o_+ \mid q)}{\pi_{old}(o_+ \mid q)} - \beta\log\frac{\pi_\theta(o_- \mid q)}{\pi_{old}(o_- \mid q)} \right) \right],
$$

其中 $o_+$ 是高奖励补全,$o_-$ 是低奖励补全。当 $G = 2$ 时,这种对比结构是显式的。当 $G = 16$ 时,同一信号存在,但被冗余样本对稀释。

**2-GRPO 的算力节省**

- $G = 2$ 对比 $G = 16$:生成计算量减少 8 倍。
- 生成通常是瓶颈(占墙上时间(wall-clock)的 60–80%)。
- 端到端总训练加速:约 4–6 倍。
- 在 GSM8K、MATH 和代码基准上无精度损失。

**2-GRPO 在 TRL 中**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    num_generations=2,                  # 关键改动 —— 仅两次采样
    loss_type="grpo",                   # 标准 GRPO 损失即可
    epsilon=0.2,
    # 当 G=2 时,batch size 必须至少为 2 * num_prompts_per_step
    per_device_train_batch_size=2,
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
)
```

**2-GRPO 的注意事项**

当 $G = 2$ 时,优势归一化仅在两个值上进行,因此归一化后的优势总是 $\{+1,\ -1\}$(若奖励相等则为 $\{0,\ 0\}$)。这意味着梯度幅度固定,与奖励差距无关。对于奖励差异的幅度重要的任务(如部分计分),更大的 $G$ 可能仍有益。

### 7.5.6 SAPO —— 软自适应策略优化(Soft Adaptive Policy Optimization)

**硬截断的脆弱性**

PPO 风格的截断制造了一个不连续的梯度:截断带外梯度为零,带内非零。这种「悬崖边缘」可能在边界附近引起不稳定,并使信任域(trust region)对 $\epsilon$ 的选择敏感。

SAPO [188] 用一个平滑的、温度控制的门函数(gate function)替换硬截断。SAPO 用一个平滑的替代项替换 $\min(\rho A,\ \text{clip}(\rho, \cdot)A)$ 目标:

$$
\mathcal{L}_{SAPO}(\rho, A) = \begin{cases} -A \cdot \sigma\!\left(\dfrac{\rho - 1}{\tau_+}\right) \cdot \rho & \text{若 } A > 0 \\[6pt] -A \cdot \sigma\!\left(\dfrac{1 - \rho}{\tau_-}\right) \cdot \rho & \text{若 } A \le 0 \end{cases}
$$

其中 $\sigma$ 是 sigmoid 函数,$\tau_+,\ \tau_-$ 是非对称温度参数。较高温度产生较软的门(更多探索);较低温度趋近硬截断。

**SAPO 温度直觉**

- $\tau_+ = 1.0$:对正优势采用中等门(允许探索)。
- $\tau_- = 1.05$:对负优势采用稍软的门(避免过度抑制)。
- 当 $\tau \to 0$:恢复硬 PPO 截断。
- 当 $\tau \to \infty$:恢复无截断的策略梯度。

**SAPO 在 TRL 中**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    loss_type="sapo",
    sapo_temperature_pos=1.0,           # 正优势的 tau_+
    sapo_temperature_neg=1.05,          # 负优势的 tau_-
    num_generations=8,
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
)
```

### 7.5.7 TIS 与 MIS —— 截断与掩码重要性采样(Truncated and Masked Importance Sampling)

**隐性的 vLLM 概率不匹配**

使用 vLLM 进行快速生成时,vLLM 返回的对数概率与训练前向传递中计算的不同 [189]。这不是 bug——它源于不同的 CUDA 内核、不同的浮点精度,以及不同的注意力实现(如 FlashAttention 与 PagedAttention)。这种不匹配会悄悄破坏同策略假设:用于计算重要性比率的「旧策略」概率是错的,导致有偏的梯度估计。

**截断重要性采样(Truncated Importance Sampling, TIS)**

TIS 通过将梯度乘以一个截断校正因子来校正偏差:

$$
w_{TIS}(o_i) = \min\!\left(C,\ \frac{\pi_{train}(o_i \mid q)}{\pi_{vllm}(o_i \mid q)}\right),
$$

其中 $\pi_{train}$ 是训练前向传递给出的概率,$\pi_{vllm}$ 是 vLLM 报告的概率。在 $C$ 处的截断防止极端校正破坏训练稳定性。

**掩码重要性采样(Masked Importance Sampling, MIS)**

MIS 采取更硬的做法:对任何校正比率超过阈值 $C$ 的序列,将其梯度置零:

$$
w_{MIS}(o_i) = \mathbb{1}\!\left[\frac{\pi_{train}(o_i \mid q)}{\pi_{vllm}(o_i \mid q)} \le C\right].
$$

这更保守,但避免了大的(即便已截断)校正权重的风险。

**序列级 vs Token 级 IS**

TIS 与 MIS 都可在 token 级或序列级应用:

- **序列级**:将比率计算为所有 token 上的几何均值(如 GSPO)。理论正确但方差更高。
- **Token 级**:为每个 token 计算单独的比率。有偏(每 token 校正的乘积并非序列校正)但方差更低。

**TIS 与 MIS 在 TRL 中**

```python
from trl import GRPOConfig, GRPOTrainer

# vLLM 概率不匹配的截断 IS 校正
config_tis = GRPOConfig(
    use_vllm=True,
    vllm_importance_sampling_correction=True,
    vllm_importance_sampling_mode="sequence_truncate",   # TIS
    vllm_importance_sampling_cap=5.0,                    # C 阈值
)

# 掩码 IS 校正
config_mis = GRPOConfig(
    use_vllm=True,
    vllm_importance_sampling_correction=True,
    vllm_importance_sampling_mode="sequence_mask",       # MIS
    vllm_importance_sampling_cap=3.0,
)
```

**何时使用 TIS/MIS**

- 在使用 vLLM 生成时始终考虑启用。
- 当不匹配较小时(同一模型、不同精度)优先用 TIS。
- 当不匹配较大或不可预测时优先用 MIS。
- 序列级 IS 在理论上更受推荐;token 级是实践上的折中。

### 7.5.8 VESPO —— 变分序列级软策略优化(Variational Sequence-Level Soft Policy Optimization)

**有原则的奖励重塑**

大多数 GRPO 变体启发式地修改截断机制。VESPO 从变分推断框架推导出一个有原则的奖励重塑核(reward-reshaping kernel),将策略优化视为近似后验推断。

VESPO [190] 推导出一个结果是光滑的、非对称的核,并自然地处理异步或异策略训练中的陈旧性(staleness)。VESPO 从变分目标为每条轨迹 $\tau$ 推导出加权函数 $W(\tau)$。最终梯度权重形如:

$$
g(\tau) = W(\tau)^{k} \cdot \exp\!\big(\lambda(1 - W(\tau))\big),
$$

其中 $W(\tau) = \pi_\theta(\tau)/\pi_{old}(\tau)$ 是序列级重要性权重,$k$ 控制加权的锐度,$\lambda$ 控制对陈旧(低权重)轨迹的指数衰减。这个核:

- 处处光滑(在截断边界没有不连续梯度)。
- 通过指数项自然地降权陈旧轨迹($W \ll 1$)。
- 是非对称的:高权重轨迹($W > 1$)与低权重轨迹的处理方式不同。

**VESPO 在 TRL 中**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    loss_type="vespo",
    vespo_k_pos=2.0,                    # 锐度指数(正优势)
    vespo_lambda_pos=3.0,               # 陈旧衰减(正优势)
    num_generations=8,
    steps_per_generation=2,             # 异策略;VESPO 处理陈旧性
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
)
```

### 7.5.9 DPPO —— 直接策略散度策略优化(Direct Policy Divergence Policy Optimization)

**比率截断的问题**

PPO 的比率截断是约束新旧策略之间 KL 散度的代理。但该代理并不完美:截断过度惩罚低概率 token(概率上一个小的绝对变化对应大的比率变化),又对高概率 token 惩罚不足(大的绝对变化对应小的比率变化)。DPPO [191] 用直接的散度估计替换比率截断。

DPPO 直接使用新旧策略分布之间的全变差距离(Total Variation, TV)或 KL 散度来计算信任域约束:

$$
\mathcal{L}_{DPPO} = -\mathbb{E}\!\left[ \hat{A} \cdot \pi_\theta(o \mid q) \cdot \mathbb{1}\!\left[D(\pi_\theta \,\|\, \pi_{old}) \le \delta\right] \right],
$$

其中 $D$ 是所选的散度度量。实践中,DPPO 用 token 级二元或 top-k 掩码来近似:

- `binary_tv`:掩码满足 $|\pi_\theta - \pi_{old}| > \delta$ 的 token。
- `binary_kl`:掩码满足 $\pi_\theta \log(\pi_\theta / \pi_{old}) > \delta$ 的 token。
- `topk_tv`:仅保留按 TV 贡献最大的前 k 个 token。
- `topk_kl`:仅保留按 KL 贡献最大的前 k 个 token。

**DPPO —— 概念实现**

DPPO 尚未作为 TRL 内置训练器提供。一个自定义实现会使用 `GRPOTrainer` 并配以修改后的损失,该损失基于分布散度(TV 或 KL)而非标准概率比率进行截断:

```python
# 伪代码:DPPO 需要一个自定义训练器子类
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    num_generations=8,
    beta=0.04,
)
# 覆盖损失计算,使用分布性截断:
# 当 TV(pi_new || pi_old) > delta 时截断,而非
# 当 pi_new/pi_old 超出 [1-eps, 1+eps] 时
```

**DPPO 处于研究阶段**

DPPO 是一项近期的研究贡献,尚未集成进主流 RL 库。当观察到标准比率截断失效时(例如在 token 概率分布高度偏斜的任务上)它最为有用。

### 7.5.10 ScaleRL 与 CISPO

**RL 的扩展定律(scaling laws)**

ScaleRL 论文 [192] 系统性地研究了是什么让 LLM 的 RL 训练能高效扩展。关键发现是:两项修改——批次级奖励缩放和 DAPO 风格的 token 级损失——共同解锁了大规模下的强劲性能,而单独任一项都不足以奏效。CISPO(Cipped IS Policy Optimization,截断 IS 策略优化)是所得算法。

**批次级奖励缩放**

标准 GRPO 在单个提示的 $G$ 个补全组内归一化奖励。CISPO 跨整个批次归一化奖励:

$$
\hat{A}_i = \frac{r_i - \mu_{batch}}{\sigma_{batch} + \epsilon},
$$

其中 $\mu_{batch}$ 与 $\sigma_{batch}$ 在当前训练批次的所有奖励上计算。这提供了更稳定的基线,并防止任何单一提示主导梯度。

**CISPO 损失**

CISPO 将批次级缩放与 DAPO 的 token 级损失聚合及非对称截断相结合:

$$
\mathcal{L}_{CISPO} = -\frac{1}{\sum_{i,t} m_{i,t}}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} m_{i,t} \cdot \min\left( \rho_{i,t}\hat{A}_i,\ \text{clip}_{DAPO}(\rho_{i,t}, \hat{A}_i)\hat{A}_i \right),
$$

其中 $m_{i,t}$ 是超长过滤掩码。

**CISPO 在 TRL 中**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    loss_type="cispo",
    scale_rewards="batch",              # 批次级奖励归一化
    mask_truncated_completions=True,
    epsilon=0.2,
    epsilon_high=5.0,                   # CISPO(ScaleRL 论文)的 epsilon_max
    num_generations=8,
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
)
```

**ScaleRL 关键发现**

1. 仅批次级奖励缩放:适度改进。
2. 仅 token 级损失:适度改进。
3. 两者结合:协同增效——显著优于任一单独使用。
4. 更大的批次大小从批次级缩放中获益更多。
5. CISPO 是大规模 RL 训练的推荐默认选择。

### 7.5.11 GDPO —— 组奖励解耦策略优化(Group Reward-Decoupled Policy Optimization)

**多奖励坍缩问题**

在多目标 RL 中(例如同时优化正确性和格式),标准 GRPO 归一化合并后的奖励。如果某个奖励的方差远高于另一个,它就主导归一化后的优势,从而实际上忽略了另一个奖励。这就是优势坍缩(advantage collapse):低方差奖励贡献近乎零的梯度。GDPO [193] 在聚合前对每个奖励独立归一化。

其核心机制在聚合前对每个奖励独立归一化:

$$
\hat{A}_n^{(i)} = \frac{r_n^{(i)} - \mu_n}{\sigma_n + \epsilon}, \qquad \hat{A}^{(i)} = \sum_{n=1}^{N} w_n\, \hat{A}_n^{(i)},
$$

其中 $r_n^{(i)}$ 是补全 $i$ 的第 $n$ 个奖励,$\mu_n$ 与 $\sigma_n$ 是组内第 $n$ 个奖励的均值与标准差,$w_n$ 是用户指定的权重。

**GDPO 与标准多奖励 GRPO 的对比**

- 标准:$\hat{A}^{(i)} = \dfrac{\sum_n w_n r_n^{(i)} - \mu_{combined}}{\sigma_{combined}}$。高方差奖励主导。
- GDPO:分别归一化每个奖励,再合并。每个奖励按其权重 $w_n$ 成比例地贡献。
- GDPO 在奖励具有非常不同的尺度或方差时不可或缺。

**GDPO 在 TRL 中**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    multi_objective_aggregation="normalize_then_sum",
    reward_weights=[1.0, 0.5],          # [正确性, 格式] 的权重
    num_generations=8,
)


def correctness_reward(completions, **kwargs):
    return [1.0 if is_correct(c) else 0.0 for c in completions]


def format_reward(completions, **kwargs):
    return [0.1 if has_good_format(c) else 0.0 for c in completions]


trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward, format_reward],
    args=config,
    train_dataset=dataset,
)
```

### 7.5.12 GOPO —— 组序数策略优化(Group Ordinal Policy Optimization)

GOPO [194] 出发于一个简单的观察:奖励模型(reward model, RM)是用成对比较训练的(「A 是否比 B 好?」),因此只有其输出的排名顺序(rank order)是可信的——原始数值分数本身不携带固有意义。然而 GRPO 把这些原始量级直接喂入优势计算。对于具有不可验证奖励的任务——摘要、开放式聊天、指令跟随——这种不匹配引入了噪声,因为 0.6 个奖励点的差距在输出空间的一处可能反映真实质量,在另一处却毫无意义。

**关键洞见**:完全丢弃奖励量级,只使用组内奖励的序数排名。

**算法**:给定一组 $N$ 个回复 $\{o_1, \dots, o_N\}$,其奖励为 $\{r_1, \dots, r_N\}$:

1. 按奖励对回复排序:赋予排名 $\text{rank}(o_i) \in \{1, \dots, N\}$($1$ = 最差,$N$ = 最好)。
2. 用基于排名的分数替换原始优势:

$$
\hat{A}_i^{GOPO} = f\!\left(\frac{\text{rank}(o_i)}{N}\right) \quad (7.2)
$$

其中 $f$ 是单调变换(例如线性映射到 $[-1, 1]$,或分位数归一化)。

3. 用基于排名的优势应用 PPO 风格的截断目标。

**与 GRPO 的对比:**

| 方面 | GRPO | GOPO |
|---|---|---|
| 优势信号 | $\hat{A}_i = (r_i - \mu)/\sigma$(使用量级) | $\hat{A}_i = f(\text{rank}_i / N)$(仅使用序数排名) |
| 对奖励尺度的敏感度 | 高——校准不良的 RM 分数扭曲优势 | 无——对单调奖励变换不变 |
| 最适用于 | 可验证奖励(二元、校准良好) | 不可验证奖励(基于 RM、量级嘈杂) |

**经验增益(相对 GRPO,在不可验证任务上)**:

- 奖励曲线(训练与留出)在整个优化过程中都高于 GRPO。
- 由独立 LLM 评估者判定的胜率在多数训练检查点上都有提升。
- 收敛明显更快——以更少的梯度步数达到 GRPO 的最终质量。
- 随着奖励模型变得更嘈杂或校准更差,这一优势进一步扩大。

**何时使用 GOPO vs. GRPO**

- **使用 GRPO**:当奖励可验证且精确时(数学正确性、代码测试通过/失败、二元信号)。量级携带有意义的信息。
- **使用 GOPO**:当奖励来自主观任务上的学习型奖励模型(有用性、风格、安全性)时。RM 的相对排序是可信的,但其绝对分数是任意的。
