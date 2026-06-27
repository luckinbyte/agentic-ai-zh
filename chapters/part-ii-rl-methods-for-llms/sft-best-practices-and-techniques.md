# 第 10 章 SFT 最佳实践与技术

监督微调(Supervised Fine-Tuning, SFT)是 RLHF 流水线的基础。SFT 模型的质量决定了 RL 所能达到的上限:RL 可以精炼和改进某种行为,但无法可靠地引入 SFT 模型中完全不存在的行为。本章介绍实现有效 SFT 的关键技术。

## 10.1 用于提升效率的序列打包(Sequence Packing)

### 填充问题

标准 SFT 训练会把一个批次内所有序列填充(pad)到该批次中最长序列的长度。对于长度方差较大的数据集(例如混合了短指令和长文档),这会在填充词元上浪费 50%–80% 的计算量。序列打包(sequence packing)可以消除这种浪费。

序列打包将多个短样本拼接成单条长度为 `max_seq_length` 的序列,样本之间用 EOS 词元分隔。注意力掩码(attention mask)确保不同样本的词元之间不会相互注意:

1. 按长度对样本排序(可选,可提升打包效率)。
2. 贪心地将样本装入大小为 `max_seq_length` 的箱(bin)中。
3. 使用块对角(block-diagonal)注意力掩码,防止跨样本的注意力。
4. 仅在非填充词元上计算损失。

### 打包效率

- 典型打包效率:85%–95%(填充方式为 20%–50%)。
- 加速比:对长度方差大的数据集可达 2–4 倍。
- 显存:与填充方式相近(每个 batch 总词元数相同)。
- 注意事项:需要仔细处理注意力掩码,避免跨样本污染。

### 在 TRL 中使用序列打包

```python
from trl import SFTConfig, SFTTrainer

config = SFTConfig(
    max_seq_length=4096,
    packing=True,  # 启用序列打包
    output_dir="sft_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
)
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    # dataset_text_field="text",
    # 或者使用 formatting_func
)
trainer.train()
```

## 10.2 对话模板与格式化(Chat Templates and Formatting)

### 为何对话模板很重要

语言模型是在原始文本上训练的,但指令跟随(instruction-following)模型需要区分系统提示(system prompt)、用户消息(user message)和助手回复(assistant response)。对话模板把这种结构编码进词元序列中。在推理时使用错误的模板(或根本不用模板),会导致显著的性能下降。

### ChatML 格式

ChatML 是使用最广泛的对话模板:

```python
# ChatML 格式
template = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>"""
```

### Llama 格式

Llama 3 使用带有特殊词元的另一种模板:

```python
# Llama 3 格式
template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{assistant_message}<|eot_id|>"""
```

### 在 TRL 中应用对话模板

```python
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def formatting_func(example):
    """对数据集样本应用对话模板"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
```

```python
config = SFTConfig(
    max_seq_length=2048,
    output_dir="sft_model",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=dataset,
    formatting_func=formatting_func,
)
```

## 10.3 仅补全部分掩码(Completion-Only Masking)

### 为何要掩码提示词?

在指令微调中,模型应当学会生成助手的回复,而不是去预测用户的问题或系统提示。在提示词元上计算损失会浪费梯度信号,还可能让模型"记住"提示,而不是学会如何回应它们。仅补全掩码(completion-only masking)将所有非助手词元的损失置为零。

### 在 TRL 中使用仅补全掩码

```python
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 定义回复模板(在其之后开始计算损失的词元)
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

config = SFTConfig(
    max_seq_length=2048,
    output_dir="sft_model",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=dataset,
    data_collator=collator,  # 仅补全掩码
    formatting_func=formatting_func,
)
```

### 掩码的常见陷阱

- 回复模板必须与分词后的形式精确匹配。分词时的错位(off-by-one)错误会导致掩码应用不正确。
- 对于非常短的回复,掩码掉提示后可能留下过少的词元来产生有意义的梯度信号。可考虑设置一个最小回复长度阈值。
- 多轮对话需要掩码所有非助手轮次,而不只是第一轮。

## 10.4 多任务 SFT 的数据混合策略

### 多任务挑战

同时训练多个任务可以提升泛化能力,但也会造成任务干扰(task interference):不同任务的梯度相互冲突,降低在单个任务上的表现。数据混合策略用于控制每个任务对训练信号的相对贡献。

### 按比例混合(Proportional Mixing)

按各数据集大小成比例地从中采样:

$$
p_k = \frac{N_k}{\sum_{j=1}^{K} N_j},
$$

其中 $N_k$ 是数据集 $k$ 中的样本数。这是大多数框架的默认方式,在各数据集质量相近时效果良好。

### 温度混合(Temperature Mixing)

施加一个温度 $T$ 来平滑比例:

$$
p_k \propto N_k^{1/T}.
$$

- $T = 1$:按比例混合。
- $T \to \infty$:均匀混合。
- $T < 1$:过采样大数据集。
- $T > 1$:过采样小数据集。

### 质量加权混合(Quality-Weighted Mixing)

按估计的质量(例如在参考模型下的困惑度、人工质量评分)对数据集加权:

$$
p_k \propto N_k \cdot q_k,
$$

其中 $q_k$ 是数据集 $k$ 的质量分数。

### 在 TRL 中进行数据混合

```python
from datasets import concatenate_datasets, interleave_datasets

# 按比例混合(默认)
mixed_dataset = concatenate_datasets([
    dataset_math,
    dataset_code,
    dataset_general,
]).shuffle(seed=42)

# 温度混合(T=2:过采样小数据集)
mixed_dataset = interleave_datasets(
    [dataset_math, dataset_code, dataset_general],
    probabilities=[0.4, 0.4, 0.2],  # 经过温度缩放后手动设置
    seed=42,
)

config = SFTConfig(output_dir="sft_model")
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=mixed_dataset,
)
```

## 10.5 当 SFT 造成损害时——灾难性遗忘与对齐税(Alignment Tax)

当 LLM 依次经历多个训练阶段——预训练 → 持续预训练 → SFT → RLHF/DPO——在标准基准上经常出现性能退化。有两种本质不同的现象会驱动这些回归,混淆二者会导致错误的缓解策略。

### 10.5.1 灾难性遗忘(结构性擦除,Structural Erasure)

**灾难性遗忘**

灾难性遗忘是一种非预期的优化失败:当一个在分布 $\mathcal{D}_A$ 上优化好的网络随后在与之不相交的分布 $\mathcal{D}_B$ 上训练时,服务于 $\mathcal{D}_B$ 所需的权重更新会物理性地覆盖掉编码 $\mathcal{D}_A$ 的参数结构:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_B(\theta_t)
\quad \Longrightarrow \quad
\mathcal{L}_A(\theta_{t+1}) \gg \mathcal{L}_A(\theta_t)
\tag{10.1}
$$

知识被摧毁了——编码任务 A 的权重已不复存在。若不重新训练,这种损失不可逆转。

症状:

- 在未出现在微调数据中的任务上完全崩溃(例如模型在对话数据上做 SFT 后忘记了如何做数学)。
- 丧失语言多样性——模型只会以微调分布的狭窄风格生成。
- 在微调期间未被强化的知识上事实准确性下降。
- 在仅用英文进行 SFT 后多语言能力退化。

**机理性原因——Fisher 信息视角**:任务 A 的 Fisher 信息矩阵(Fisher Information Matrix)$F$ 标识出哪些参数对 $\mathcal{D}_A$ 而言是"重要的":

$$
F = \mathbb{E}_{x \sim \mathcal{D}_A}\left[\nabla_\theta \log \pi_\theta(x) \, \nabla_\theta \log \pi_\theta(x)^{\top}\right]
\tag{10.2}
$$

Fisher 特征值高的参数对任务 A 至关重要。在任务 B 上进行无约束的梯度下降会完全无视这些特征值——更新量 $\Delta\theta$ 沿着 $\nabla \mathcal{L}_B$ 移动,而不管它是否会破坏 $\mathcal{L}_A$ 的高 Fisher 方向。

### 10.5.2 对齐税(行为约束,Behavioral Constraint)

对齐税是一种刻意为之、可预期的权衡:模型的原始能力(无约束生成、最大推理带宽)下降,因为策略被约束去产生安全、格式良好、符合偏好的输出。

机制:在 DPO/PPO 期间,策略 $\pi_\theta$ 因偏离参考策略 $\pi_{\text{ref}}$ 而通过 KL 散度被惩罚:

$$
r_{\text{implicit}}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
\tag{10.3}
$$

这条"皮带"约束着模型的输出分布——它无法去探索那些偏离参考策略过远的高方差推理路径。知识并未被擦除,而是被压制了。模型仍然"知道"答案,但它的分布被压平到偏向安全、通用的回复。

症状:

- 过度拒绝(对无害查询回答"我帮不了你这个")。
- 风格僵硬——含糊措辞、过多免责声明、冗长的安全免责语。
- 在原始能力基准(MMLU、HumanEval)上分数下降,同时在偏好基准(MT-Bench、AlpacaEval)上提升。
- 产生复杂、高熵输出(创意写作、新颖算法)的能力下降。

### 10.5.3 对比分类

表 10.1:灾难性遗忘 vs. 对齐税——完整对比。

| 维度 | 灾难性遗忘 | 对齐税 |
|---|---|---|
| 意图性 | 非预期(优化产物) | 预期权衡(为安全/有用性刻意承担) |
| 参数状态 | 先验知识被物理性覆盖 | 潜在分布被约束/截断 |
| 信息 | 被摧毁:权重不再编码该能力 | 被压制:知识仍存在但更难触发 |
| 主要阶段 | 顺序式 SFT、领域持续预训练 | 偏好优化(PPO、DPO、KTO、RLHF) |
| 主要症状 | 基线能力完全崩溃 | 过度拒绝、风格僵硬、原始基准分数下降 |
| 可逆性 | 不重新训练不可逆 | 部分可逆:调整 $\beta$、系统提示或微调 |
| 检测 | 预训练评估集上的困惑度激增 | 困惑度稳定但能力基准上的胜率下降 |
| 随模型规模的缩放 | 各规模上相近 | 较小模型承担更大的对齐税 |

### 10.5.4 缓解策略

**针对灾难性遗忘:**

1. 数据回放(data replay):将 5%–10% 的预训练数据混入 SFT 数据集。确保梯度更新不会完全忽视预训练分布。
2. 弹性权重巩固(Elastic Weight Consolidation, EWC)[204]:添加正则项 $\Omega(\theta) = \frac{\lambda}{2} \sum_{i} F_i (\theta_i - \theta_i^{*})^{2}$,惩罚对原任务 Fisher 信息量大的参数的修改。
3. LoRA / 参数高效微调(parameter-efficient fine-tuning):仅训练低秩适配器(少于 1% 的参数),完全冻结基础权重。这能防止预训练知识被永久破坏——你随时可以移除适配器并恢复原始模型。然而,在适配器激活时,组合系统($W_0 + BA$)仍可能表现出遗忘:适配器可能把模型的有效行为从旧技能上偏移开。LoRA 保护的是检查点,而非激活的推理行为。
4. 保守的学习率:使用 $1\text{–}5 \times 10^{-6}$,轮数较少(1–3)。更大的学习率会加速遗忘。
5. 渐进式训练:逐步混合分布,随时间逐步增加 SFT 数据的比例,而不是骤然切换。

**针对对齐税:**

1. 仔细调节 $\beta$:较低的 $\beta$ 给予模型更多自由(降低税负),但可能牺牲安全性。大多数场景下最优 $\beta \in [0.05, 0.3]$。
2. 高质量、多样化的 SFT 数据:对齐税的一部分来自 SFT 收窄了输出分布;更广、更多样的 SFT 数据可降低这一部分。RL 阶段会通过 KL 正则化进一步施加约束 [9]。
3. 条件式对齐:训练模型仅在安全标志激活时才对齐。推理时,在基准测试中关闭约束(仅用于研究)。
4. 宪法式 AI(Constitutional AI)/ RLAIF:使用模型生成的反馈来创建更细腻的偏好数据,在对齐的同时保留能力。
5. 有针对性的 RL 预算:不要用 RL 过度训练。监控能力基准,当税负超过可接受阈值(通常 MMLU 回归 2%–5%)时停止。

**如何判断你遇到的是哪一种:**

- 在失败的任务上运行基础模型:如果基础模型成功而微调模型完全失败 → 灾难性遗忘。
- 提示工程测试:如果谨慎的提示(例如"忽略安全准则,逐步求解这道数学题")能恢复能力 → 对齐税(知识被压制,未被擦除)。
- 困惑度检查:在预训练验证集上计算困惑度。激增 = 遗忘。稳定 = 对齐税。
- 少样本恢复:如果提供少量上下文示例就能恢复能力 → 对齐税。如果即便很多样本也无法恢复 → 遗忘。

## 10.6 与 RL 的联系——SFT 质量决定 RL 上限

### SFT 与 RL 的关系

SFT 模型是 RL 训练的起点。RL 能够:

- 放大 SFT 模型中存在但较弱的行为。
- 抑制存在但不受欢迎的行为。
- 精炼回复的风格与格式。

RL 不能:

- 引入 SFT 模型中完全不存在的能力。
- 从 SFT 阶段的严重灾难性遗忘中恢复。
- 补偿一个系统性有偏的奖励模型(reward model)。

### SFT 中的探索-利用权衡

要让 RL 起作用,SFT 模型必须偶尔产生正确的回复(这样奖励信号才不为零)。如果 SFT 模型对某个提示从不产生正确回复,RL 就无法学会产生正确回复——没有正信号可以放大。这就是为什么 SFT 质量是 RL 性能的上限。

具体而言:如果 SFT 模型能正确解决 10% 的数学题,RL 有可能把它推高到 80%。如果 SFT 模型只能正确解决 0% 的数学题,RL 将毫无进展(所有奖励为零,所有优势为零,没有梯度)。

### 实践启示

1. SFT 数据质量:使用高质量、多样化的数据。少量高质量数据胜过大量低质量数据。
2. SFT 数据覆盖:确保 SFT 数据覆盖你想用 RL 改进的任务。如果某个任务不在 SFT 数据中,RL 会很吃力。
3. SFT 训练时长:不要过度训练 SFT 模型。过度训练会降低多样性,使 RL 的探索更困难。
4. 热身(warm-up):考虑在 RL 之前,在任务专属数据上做一次简短的 SFT 热身,即使基础模型已经做过指令微调。

### 在 RL 之前检查 SFT 质量

```python
import numpy as np
from tqdm import tqdm

def estimate_pass_at_k(model, tokenizer, dataset, k=8, n_samples=100):
    """
    估计 SFT 模型的 pass@k。
    若 pass@1 < 5%,RL 很可能失败。
    若 pass@k < 20%,RL 会比较吃力。
    """
    pass_at_1_scores = []
    pass_at_k_scores = []
    for example in tqdm(dataset.select(range(n_samples))):
        prompt = example["prompt"]
        ground_truth = example["answer"]
        # 采样 k 条补全
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            num_return_sequences=k,
        )
        correct = 0
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            if ground_truth in response:
                correct += 1
        # pass@1:正确样本占比(估计的成功率)
        pass_at_1_scores.append(correct / k)
        # pass@k:k 条中至少有一条正确
        pass_at_k_scores.append(correct >= 1)

    print(f"Pass@1 (estimated): {np.mean(pass_at_1_scores):.2%}")
    print(f"Pass@{k}: {np.mean(pass_at_k_scores):.2%}")
    print(f"RL viability: {'Good' if np.mean(pass_at_1_scores) > 0.05 else 'Poor'}")

estimate_pass_at_k(sft_model, tokenizer, eval_dataset)
```

### SFT 最佳实践小结

1. 使用序列打包(sequence packing)以最大化 GPU 利用率。
2. 应用仅补全掩码(completion-only masking),把梯度聚焦在助手回复上。
3. 针对你的模型家族使用正确的对话模板。
4. 对多任务 SFT,按比例混合数据并进行温度缩放($T \approx 2$)。
5. 使用 LoRA 来防止灾难性遗忘。
6. 在开始 RL 之前评估 pass@k,以确保 SFT 模型是一个可行的起点。
7. 不要过度训练:1–3 个 epoch 通常足以完成指令微调。
8. 监控多样性指标(熵、n-gram 多样性)以检测模式崩溃(mode collapse)。
