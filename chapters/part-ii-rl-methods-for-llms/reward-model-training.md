# 第 9 章 奖励模型训练(Reward Model Training)

奖励模型(reward model)是连接人类偏好与 RL 训练信号的桥梁。一个训练良好的奖励模型对于成功的 RLHF 至关重要;而训练不良的模型则会导致奖励黑客(reward hacking)与行为错位(misalignment)。本章涵盖奖励模型的理论基础、实践训练技巧与架构选择。

## 9.1 Bradley-Terry 模型 —— 完整推导

Bradley-Terry 模型(Bradley-Terry model)[198] 是成对偏好学习(pairwise preference learning)的标准概率框架。给定提示 $q$ 的两个回复 $y_1$ 和 $y_2$,该模型假设:

$$
P(y_1 \succ y_2 \mid q) = \sigma\!\left(r(y_1, q) - r(y_2, q)\right) = \frac{e^{r(y_1, q)}}{e^{r(y_1, q)} + e^{r(y_2, q)}}, \tag{9.1}
$$

其中 $r : \mathcal{Y} \times \mathcal{Q} \to \mathbb{R}$ 是标量奖励函数,$\sigma$ 是 sigmoid 函数。

**最大似然估计(Maximum Likelihood Estimation)**

给定由偏好对组成的数据集 $D = \{(q^{(k)}, y_w^{(k)}, y_l^{(k)})\}_{k=1}^{N}$,MLE 目标为:

$$
\mathcal{L}_{BT}(\phi) = -\frac{1}{N} \sum_{k=1}^{N} \log \sigma\!\left( r_\phi(y_w^{(k)}, q^{(k)}) - r_\phi(y_l^{(k)}, q^{(k)}) \right), \tag{9.2}
$$

其中 $r_\phi$ 是以 $\phi$ 为参数的神经网络。这是一个二元交叉熵损失(binary cross-entropy loss),其中“正类”即被偏好的回复。

**Bradley-Terry 假设**

1. 偏好具有传递性:若 $y_1 \succ y_2$ 且 $y_2 \succ y_3$,则 $y_1 \succ y_3$。
2. 偏好由标量奖励决定(不存在多维偏好)。
3. 偏好概率仅取决于奖励之差。
4. 各偏好对之间相互独立(无标注者效应)。

这些假设在实践中常被违背,从而催生了诸如用于排序的 Plackett-Luce 模型以及多维奖励模型等扩展。

**间隔损失扩展(Margin Loss Extension)**

一种常见的扩展是在获胜奖励与失败奖励之间加入间隔(margin)$m$,以确保二者之间至少存在一个最小间距:

$$
\mathcal{L}_{\text{margin}} = -\frac{1}{N} \sum_{k=1}^{N} \log \sigma\!\left( r_\phi(y_w^{(k)}, q^{(k)}) - r_\phi(y_l^{(k)}, q^{(k)}) - m \right). \tag{9.3}
$$

## 9.2 奖励模型架构

**LLM 上的分类头(Classification Head on LLM)**

标准的奖励模型架构取一个预训练 LLM,将其语言建模头(将隐状态映射到词表 logits)替换为一个标量回归头(将最终的隐状态映射到单个奖励值)。

其架构如下:

1. **主干(Backbone)**:一个预训练 LLM(例如 Llama、Mistral),将提示-回复对编码为一系列隐状态。
2. **池化(Pooling)**:提取最后一个 token 位置的隐状态(对于 decoder-only 模型)或 `[CLS]` token 的隐状态(对于 encoder 模型)。
3. **回归头(Regression head)**:一个线性层 $W \in \mathbb{R}^{d \times 1}$,将池化后的隐状态映射为一个标量奖励。

**TRL 中的奖励模型训练**

```python
from trl import RewardConfig, RewardTrainer
from transformers import AutoModelForSequenceClassification

# 加载带标量头的模型(num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    num_labels=1,
)
config = RewardConfig(
    output_dir="reward_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    # 奖励中心化正则项
    center_rewards_coefficient=0.01,
)
trainer = RewardTrainer(
    model=model,
    args=config,
    train_dataset=dataset,  # 数据集必须含 chosen/rejected 列
)
trainer.train()
```

## 9.3 奖励模型训练技巧

**奖励中心化(Reward Centering)**

原始奖励模型的输出可能具有任意的尺度与偏移。对奖励做中心化(减去均值)可稳定 RL 训练:

$$
r_{\text{centered}}(y, q) = r_\phi(y, q) - \mathbb{E}_{y' \sim \pi_\theta}\!\left[ r_\phi(y', q) \right]. \tag{9.4}
$$

在 TRL 中,这通过 `center_rewards_coefficient` 参数实现——它在奖励模型损失中加入一个正则项,惩罚非零均值的奖励。

**长度偏差校正(Length Bias Correction)**

已知奖励模型存在长度偏差(length bias):无论质量如何,它们倾向于给更长的回复打更高的分。这可通过以下方式校正:

1. **长度归一化(Length normalisation)**:将奖励除以回复长度。
2. **长度受控训练(Length-controlled training)**:将长度作为一个特征纳入,训练模型使其对长度不变(length-invariant)。
3. **校准(Calibration)**:通过事后回归去除长度效应。

**间隔损失(Margin Losses)**

向 Bradley-Terry 损失中加入间隔 $m$ 可确保奖励模型为偏好与不偏好的回复分配有显著差异的分数:

$$
\mathcal{L}_{\text{margin}} = \max\!\left( 0,\; m - (r_w - r_l) \right). \tag{9.5}
$$

## 9.4 过程奖励模型 vs 结果奖励模型

**PRM vs ORM 对比**

| 属性 | ORM | PRM |
|---|---|---|
| 奖励信号 | 仅最终答案 | 每个推理步骤 |
| 训练数据 | (prompt, answer, correct?) | (prompt, steps, step labels) |
| 标注成本 | 低 | 高 |
| 信用分配(credit assignment) | 稀疏 | 稠密 |
| 奖励黑客 | 较易被攻破 | 较难被攻破 |
| 最适用于 | 简单任务 | 多步推理 |
| 推理成本 | 低 | 高(每步都要打分) |

**何时使用 PRM**

过程奖励模型(Process Reward Models, PRM)在以下情形最有价值:

- 任务需要多步推理(数学、代码、逻辑)。
- 最终答案是二元的(正确/错误),但中间步骤的质量参差不齐。
- 你希望将奖励模型用于搜索(例如带步骤分数的束搜索(beam search))。
- 你能够获取步骤级标注(或可自动生成)。

对于简单任务(情感、毒性、事实性),结果奖励模型(Outcome Reward Models, ORM)已经足够,且成本低得多。

**LLM 的 RLHF 中的 PBRS**

- **原始奖励**:二元正确性(若最终答案正确则为 1,否则为 0)——对多步推理而言信号极度稀疏。
- **势函数(Potential function)**:$\Phi(s) =$ 来自某个验证器(verifier)的部分学分(例如,中间推理步骤中逻辑有效的步骤所占比例)。
- **塑形奖励(Shaped reward)**:智能体(agent)每完成一个有效推理步骤即可获得增量信号,同时保留如下保证:最优策略仍以最大化最终答案正确性为目标。

实践实现包括:

- 为思维链(chain-of-thought)中每个步骤打分的过程奖励模型(PRM)
- 代码生成中的中间编译检查
- 多部分答案的部分匹配分数

这是基于势函数的奖励塑形(Potential-Based Reward Shaping, PBRS)[173] 在 LLM 场景中的直接应用——塑形奖励保留最优策略的理论保证使得 PRM 成为推理任务中获取稠密奖励的一条有原则的途径。

**自动 PRM 标注**

步骤级标注可借助以下方式自动生成:

1. **蒙特卡洛回放(Monte Carlo rollouts)**:对每个中间步骤,采样多个续写,并以能到达正确答案的占比作为该步骤的奖励。
2. **LLM 作为评判者(LLM-as-judge)**:使用一个强 LLM 来评估每个步骤。
3. **形式化验证(Formal verification)**:对于数学/代码,使用验证器检查每个步骤。

## 9.5 RLVR 的基于规则的奖励

基于可验证奖励的强化学习(Reinforcement Learning from Verifiable Rewards, RLVR)使用确定性的、基于规则的奖励函数,而非学习得到的奖励模型。这大幅减少了奖励黑客(尽管模型仍可能利用格式技巧、边界用例或测试记忆),也是 DeepSeek-R1 [15] 所采用的方法。

**TRL 中的基于规则的奖励函数**

```python
import re

def format_reward(completions, **kwargs):
    """Reward for using <think>...</think><answer>...</answer> format."""
    # 对采用 <think>...</think><answer>...</answer> 格式的回复给予奖励
    rewards = []
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    for completion in completions:
        text = completion[0]["content"]
        rewards.append(1.0 if re.fullmatch(pattern, text, re.DOTALL) else 0.0)
    return rewards

def correctness_reward(completions, ground_truth, **kwargs):
    """Reward for correct final answer."""
    # 对最终答案正确给予奖励
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        text = completion[0]["content"]
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            rewards.append(1.0 if answer == gt else 0.0)
        else:
            rewards.append(0.0)
    return rewards

def code_execution_reward(completions, test_cases, **kwargs):
    """Reward for code that passes test cases."""
    # 对通过测试用例的代码给予奖励
    import subprocess, tempfile, os
    rewards = []
    for completion, tests in zip(completions, test_cases):
        code = completion[0]["content"]
        passed = 0
        for test in tests:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code + "\n" + test)
                fname = f.name
        try:
            result = subprocess.run(
                ["python", fname], capture_output=True,
                timeout=5, text=True
            )
            passed += int(result.returncode == 0)
        except Exception:
            pass
        finally:
            os.unlink(fname)
        rewards.append(passed / len(tests))
    return rewards
```

**基于规则的奖励的陷阱(Rule-Based Reward Pitfalls)**

- **格式钻营(Format gaming)**:模型学会生成正确的格式却没有正确的内容。务必将格式奖励与正确性奖励结合使用。
- **测试用例泄漏(Test case leakage)**:若测试用例出现在训练数据中,模型会将其记忆下来。
- **超时利用(Timeout exploitation)**:模型可能生成会超时的代码(从而规避失败判定)。应使用严格的超时设置,并对超时显式地施加惩罚。
- **奖励稀疏(Reward sparsity)**:二元奖励(0/1)对于复杂任务可能过于稀疏。可考虑部分学分(partial credit)或中间奖励。

## 9.6 多目标奖励 —— 组合策略

当使用多个奖励信号进行训练时,组合策略会显著影响最终策略。

**多奖励组合策略(Multi-Reward Combination Strategies)**

1. **加权和(Weighted sum)**:$r = \sum_n w_n r_n$。简单,但对尺度敏感。
2. **归一化后求和(Normalise then sum, GDPO)**:将组内每个奖励归一化为零均值、单位方差,再加权求和。尺度不变(scale-invariant)。
3. **字典序(Lexicographic)**:按优先级顺序优化各奖励;仅当高优先级奖励并列时才考虑低优先级奖励。
4. **约束式(Constrained)**:在次级奖励的约束下最大化主奖励。
5. **帕累托(Pareto)**:维护一个策略的帕累托前沿(Pareto front),并基于偏好进行选择。

**TRL 中的多奖励训练**

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    # GDPO:独立地归一化每个奖励
    multi_objective_aggregation="normalize_then_sum",
    reward_weights=[1.0, 0.3, 0.1],  # correctness, format, length
    num_generations=8,
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        correctness_reward,
        format_reward,
        length_penalty_reward,
    ],
    args=config,
    train_dataset=dataset,
)
```

## 9.7 列表式基于排序的奖励

Bradley-Terry 模型处理的是成对偏好($y_w \succ y_l$),但许多实际场景需要同时对多个回复进行排序。列表式奖励模型(listwise reward models)从完整的排序中学习,提供更丰富的训练信号,并支持更好的校准。

**动机:超越成对(Why Listwise?)**

- **更丰富的信号**:对 $K$ 个回复的一个排序蕴含 $\binom{K}{2}$ 次隐式的成对比较,同时还能捕捉相对间距(排名 1 比排名 3 好多少)。
- **更好的校准**:成对 BT 模型只学到奖励之差;列表式模型学到奖励的绝对尺度。
- **与 GRPO 天然契合**:GRPO 为每个提示生成 $N$ 个回复并对它们排序——列表式奖励与这一工作流直接对齐。
- **标注效率**:对 5 个回复排序,比独立标注全部 10 种可能的偏好对更快。

**Plackett-Luce 模型**

Plackett-Luce(PL)模型 [199] 是 Bradley-Terry 向完整排序的标准扩展。给定 $K$ 个回复 $y_1, \dots, y_K$ 及排序 $\pi$(其中 $\pi(1)$ 为最佳):

$$
P(\pi \mid q) = \prod_{i=1}^{K} \frac{e^{r_\phi(y_{\pi(i)}, q)}}{\sum_{j=i}^{K} e^{r_\phi(y_{\pi(j)}, q)}}. \tag{9.6}
$$

**直觉**:依次选出剩余项中最佳者。在每一步,选中项 $\pi(i)$ 的概率是在剩余项上做 softmax。

**损失函数**:

$$
\mathcal{L}_{PL}(\phi) = -\frac{1}{|D|} \sum_{(q, \pi) \in D} \sum_{i=1}^{K-1} \left[ r_\phi(y_{\pi(i)}, q) - \log \sum_{j=i}^{K} e^{r_\phi(y_{\pi(j)}, q)} \right]. \tag{9.7}
$$

**Plackett-Luce 退化为 Bradley-Terry**

当 $K = 2$ 时,PL 模型给出:$P(y_1 \succ y_2) = \dfrac{e^{r(y_1)}}{e^{r(y_1)} + e^{r(y_2)}} = \sigma(r(y_1) - r(y_2))$ —— 恰好就是 Bradley-Terry 模型。PL 是其严格推广。

**ListMLE 与基于排序的损失(ListMLE and Rank-Based Losses)**

列表式损失函数:

- **ListMLE [200]**:直接最大化真实排序(ground-truth ranking)的 PL 似然。简单而有效。
- **ListNet [201]**:最小化模型的 top-1 概率分布与真实分布之间的 KL 散度(KL divergence):

$$
\mathcal{L}_{\text{ListNet}} = -\sum_{i=1}^{K} P_{\text{true}}(y_i \text{ 为最佳}) \cdot \log P_{\text{model}}(y_i \text{ 为最佳}), \tag{9.8}
$$

其中 $P_{\text{model}}(y_i \text{ 为最佳}) = \dfrac{e^{r_\phi(y_i)}}{\sum_j e^{r_\phi(y_j)}}$。

- **LambdaRank [202]**:以排序指标(例如 NDCG)的变化量对成对梯度加权。当排序质量在顶部更为重要时很有用。
- **RankNet [203]**:对所有成对组合求和的成对交叉熵——等价于从排序中抽取的全部 $\binom{K}{2}$ 个偏好对上的 BT。

**用于 GRPO 与拒绝采样的列表式奖励**

**与 GRPO 集成(Integration with GRPO)**

GRPO 天然产生带排序的组:对每个提示,$N$ 个回复被打分并排序。列表式奖励模型可直接在这些排序上训练:

1. **生成(Generate)**:从策略中为每个提示采样 $N = 8$ 个回复。
2. **排序(Rank)**:使用一个已有的奖励模型(或人类标注者)产生完整排序 $\pi$。
3. **训练列表式 RM(Train listwise RM)**:在 $(q, \pi)$ 元组上优化 PL 损失。
4. **在 GRPO 中使用(Use in GRPO)**:列表式 RM 为每个回复分配标量奖励 $r(y_i, q)$;GRPO 计算优势(advantage)为 $\hat{A}_i = (r_i - \mu) / \sigma$。

**相较成对的优势**:列表式 RM 同时看到全部 $N$ 个回复,从而学到排名 1 应当比排名 $N$ 拥有高得多的奖励(而非仅仅“比另一个回复略好”)。

**实践考量(Practical Considerations)**

**列表式训练的挑战(Listwise Training Challenges)**

- **标注成本(Annotation cost)**:完整排序代价高昂。部分排序(8 个中取前 3)能在质量损失极小的情况下降低成本。
- **并列(Ties)**:真实排序常常存在并列。可使用支持并列的 Plackett-Luce 扩展:为并列项分配相等的概率质量。
- **位置偏差(Position bias)**:标注者倾向于偏好靠前展示的项。应随机化呈现顺序并训练去偏。
- **列表长度(List length)**:典型训练取 $K = 4$–$8$。更长的列表($K > 16$)只增加噪声而收益甚微。
- **一致性(Consistency)**:不同标注者给出的排序可能不一致。应将标注者间一致性($\kappa > 0.6$)用作质量过滤器。

**Plackett-Luce 训练代码**

```python
import torch
import torch.nn.functional as F

def plackett_luce_loss(rewards, rankings):
    """
    Args:
        rewards: (batch, K) - 预测的 K 个回复的标量奖励
        rankings: (batch, K) - 真实排序索引(0 = 最佳)
    Returns:
        scalar loss
    """
    batch_size, K = rewards.shape
    # 按真实排序顺序重排奖励
    sorted_rewards = torch.gather(rewards, 1, rankings)  # (batch, K)
    # PL 对数似然:在各位置上求和
    loss = 0.0
    for i in range(K - 1):
        # 在剩余项(位置 i 到 K)上做 Log-softmax
        remaining = sorted_rewards[:, i:]  # (batch, K-i)
        log_probs = remaining[:, 0] - torch.logsumexp(remaining, dim=1)
        loss -= log_probs.mean()
    return loss / (K - 1)

# 示例:每个提示 8 个回复,由标注者排序
rewards = reward_model(responses)  # (batch, 8)
rankings = torch.argsort(human_scores, descending=True)  # 最佳在前
loss = plackett_luce_loss(rewards, rankings)
loss.backward()
```
