# 第 6 章 DPO —— 直接偏好优化(Direct Preference Optimization)

## 6.1 动机

PPO(Proximal Policy Optimization,近端策略优化)需要同时在内存中驻留 4 个模型(策略模型、参考模型、奖励模型、价值头),其 RL(强化学习,Reinforcement Learning)基础设施极为复杂,而且训练的稳定性是出了名的差。DPO(Direct Preference Optimization,直接偏好优化)[10] 提出了这样一个问题:我们能否跳过 RL,直接从偏好数据中学习?

**核心洞见**:在 RLHF(基于人类反馈的强化学习)目标(奖励最大化 + KL 散度惩罚)下的最优策略存在**闭式解(closed-form solution)**。由此我们可以推导出一个监督学习损失,它隐式地优化同一个目标。

## 6.2 数学推导

**第 1 步**:RLHF 目标函数为

$$
\max_{\pi}\; \mathbb{E}_{x,y \sim \pi}\big[r(x, y)\big] - \beta\, D_{KL}\big[\pi \,\|\, \pi_{\text{ref}}\big]
$$

**第 2 步**:该目标的最优解为

$$
\pi^{*}(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\left( \frac{r(x, y)}{\beta} \right)
$$

**第 3 步**:重排上式,把奖励用策略表示出来

$$
r(x, y) = \beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

**第 4 步**:代入 Bradley-Terry 偏好模型 $P(y_w \succ y_l) = \sigma\big(r(y_w) - r(y_l)\big)$。注意 $Z(x)$ 项会**相互抵消**!于是得到

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,\, y_w,\, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right] \quad (6.1)
$$

### DPO 实际上在做什么

定义**隐式奖励(implicit reward)**为

$$
\hat{r}(x, y) = \beta \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\text{ref}}(y \mid x)}.
$$

DPO 最小化一个交叉熵损失,其「标签」是:被选中的(chosen)回答应当具有比被拒绝的(rejected)回答更高的隐式奖励。该间隔(margin)由 $\beta$ 控制:

- **较大的 $\beta$**:需要更大的间隔 → 策略移动更激进 → 有遗忘(已学能力)的风险;
- **较小的 $\beta$**:小间隔即可满足 → 策略贴近参考模型 → 行为保守。

参考模型起到正则化器的作用:策略必须通过展现与偏好的一致性,来「证明」任何偏离参考模型的合理性。

## 6.3 梯度分析

DPO 的梯度可以分解为

$$
\nabla_{\theta} \mathcal{L} = -\,\beta \cdot \underbrace{\sigma\big(-\hat{r}_w + \hat{r}_l\big)}_{\text{权重:模型判错时该值更大}} \cdot \Big[ \nabla_{\theta} \log \pi_{\theta}(y_w \mid x) - \nabla_{\theta} \log \pi_{\theta}(y_l \mid x) \Big] \quad (6.2)
$$

**解读**:梯度会提高被选中回答的概率、降低被拒绝回答的概率。当模型当前恰恰偏好错误答案时,该权重达到最大值 —— 它把学习资源集中到「容易混淆」的回答对上。

### 具体的 DPO 示例

**提示(prompt)**:「用给 10 岁孩子的话解释量子纠缠。」

**被选中回答($y_w$)**:「想象你有两枚魔法硬币。当你掷出其中一枚得到正面时,另一枚会瞬间变成反面,无论它们相隔多远!」

$\log \pi_{\theta}(y_w \mid x) = -15.3$,$\log \pi_{\text{ref}}(y_w \mid x) = -16.1$

**被拒绝回答($y_l$)**:「量子纠缠是这样一种现象:两个粒子相互关联,以至于无法独立地描述其中一个粒子的量子态。」

$\log \pi_{\theta}(y_l \mid x) = -12.8$,$\log \pi_{\text{ref}}(y_l \mid x) = -12.5$

**隐式奖励**:$\hat{r}_w = 0.1 \times \big((-15.3) - (-16.1)\big) = 0.08$,$\hat{r}_l = 0.1 \times \big((-12.8) - (-12.5)\big) = -0.03$

**损失输入**:$\sigma\big(0.08 - (-0.03)\big) = \sigma(0.11) = 0.527$

**损失**:$-\log(0.527) = 0.64$ —— 模型几乎分不出哪个是偏好回答,梯度会强力推动更新。

**训练后**:被选中回答的概率上升、被拒绝回答的概率下降,直到间隔稳定在 $1/(2\beta)$ 附近。

## 6.4 TRL 实现

下面给出一个使用 HuggingFace TRL 的最小可运行示例。

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Dataset format: {"prompt": str, "chosen": str, "rejected": str}
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences")

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                          # KL 正则化强度
    learning_rate=5e-7,                # 极低的学习率以保证稳定性
    loss_type="sigmoid",               # 标准 DPO 损失
    max_length=2048,                   # 最大序列长度
    max_prompt_length=1024,            # 提示的截断长度
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,     # 有效批次大小 = 16
    gradient_checkpointing=True,
    bf16=True,
    num_train_epochs=1,                # DPO 很快过拟合,只跑 1 个 epoch!
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,                    # 使用 LoRA 时,参考模型即基座模型本身(无需拷贝!)
    args=dpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    peft_config=lora_config,
)
trainer.train()

# 关键监控指标:train/rewards/chosen、train/rewards/rejected、train/rewards/margins
```

## 6.5 DPO 的完整工作机理

本节给出 DPO 的完整计算细节 —— 训练过程中在词元(token)层面究竟发生了什么。

### 6.5.1 序列级对数概率

DPO 中的核心量是:给定提示 $x$ 时,整个序列 $y = (y_1, y_2, \dots, y_T)$ 的对数概率。它被计算为逐词元对数概率之和:

$$
\log \pi_{\theta}(y \mid x) = \sum_{t=1}^{T} \log \pi_{\theta}(y_t \mid x, y_{<t}) \quad (6.3)
$$

其中每一项 $\log \pi_{\theta}(y_t \mid x, y_{<t})$ 是模型在位置 $t$ 处、对序列中真实词元 $y_t$ 的 log-softmax 输出。这与标准语言建模中使用的交叉熵损失完全相同 —— 区别只在于这里我们是**求和**而非取平均。

**关键细节**:梯度会流经 $y_w$ 和 $y_l$ 中每一个词元位置。中间词元不会被掩蔽 —— 每个词元都对序列级对数概率有所贡献。

### 6.5.2 DPO 损失的分解

从损失出发

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,\, y_w,\, y_l) \sim \mathcal{D}} \Big[ \log \sigma\big(\beta \cdot h_{\theta}(x, y_w, y_l)\big) \Big] \quad (6.4)
$$

其中「隐式奖励间隔」$h_{\theta}$ 定义为

$$
h_{\theta}(x, y_w, y_l) = \underbrace{\log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}}_{\text{被选中回答的奖励代理}} - \underbrace{\log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}}_{\text{被拒绝回答的奖励代理}} \quad (6.5)
$$

展开到词元层面

$$
h_{\theta} = \sum_{t=1}^{|y_w|} \left[ \log \pi_{\theta}(y_w^t \mid x, y_{<t}^{w}) - \log \pi_{\text{ref}}(y_w^t \mid x, y_{<t}^{w}) \right] - \sum_{t=1}^{|y_l|} \left[ \log \pi_{\theta}(y_l^t \mid x, y_{<t}^{l}) - \log \pi_{\text{ref}}(y_l^t \mid x, y_{<t}^{l}) \right] \quad (6.6)
$$

### 6.5.3 前向传播:逐步解析

对于一个训练样本 $(x, y_w, y_l)$:

1. **拼接**:形成两条序列 $[x;\, y_w]$ 和 $[x;\, y_l]$。在批次内填充(pad)至等长。
2. **前向传播(策略 $\pi_{\theta}$)**:将两条序列都过一遍模型。收集每个回答位置的 logits。
3. **抽取对数概率**:在回答的每个位置 $t$,取 $\log\,\text{softmax}(\text{logits}_t)[y_t]$ —— 即真实词元的对数概率。
4. **对词元求和**:

$$
\text{logp\_chosen} = \sum_{t \in \text{response positions}} \log \pi_{\theta}(y_w^t \mid x, y_{<t}^{w}) \quad (6.7)
$$

$$
\text{logp\_rejected} = \sum_{t \in \text{response positions}} \log \pi_{\theta}(y_l^t \mid x, y_{<t}^{l}) \quad (6.8)
$$

5. **减去参考值(预先计算或通过第二次前向传播得到)**:

$$
\text{ratio\_w} = \text{logp\_chosen} - \text{ref\_logp\_chosen} \quad (6.9)
$$

$$
\text{ratio\_l} = \text{logp\_rejected} - \text{ref\_logp\_rejected} \quad (6.10)
$$

6. **计算损失**:$\mathcal{L} = -\log \sigma\big(\beta \cdot (\text{ratio\_w} - \text{ratio\_l})\big)$
7. **反向传播**:梯度沿着第 5 → 4 → 3 → 2 步回流,以更新 $\theta$。

### 6.5.4 词元级梯度分析

**每个词元都会得到梯度吗?是的。** 在被选中序列位置 $t$ 处、关于 logits 的梯度为

$$
\frac{\partial \mathcal{L}}{\partial \text{logits}^{(w)}_t} = -\,\underbrace{\sigma\big(-\beta \cdot h_{\theta}\big)}_{\text{缩放因子}} \cdot \beta \cdot \frac{\partial \log \pi_{\theta}(y_w^t \mid \cdot)}{\partial \text{logits}^{(w)}_t} \quad (6.11)
$$

**关键洞见**:缩放因子 $\sigma(-\beta \cdot h_{\theta})$ 在两条序列的所有词元间是**共享的**。它起到了一个**自适应学习率**的作用:

- 当 $h_{\theta}$ 较小(模型无法区分被选中与被拒绝)时:缩放因子 $\approx 0.5$ —— 梯度强,激进学习;
- 当 $h_{\theta}$ 较大(模型已经偏好被选中回答)时:缩放因子 $\approx 0$ —— 梯度可忽略,不再过拟合。

**对被选中词元的作用**:概率被提高(对数概率被推高)。
**对被拒绝词元的作用**:概率被降低(对数概率被压低)。
**相对于参考模型**:只有相对 $\pi_{\text{ref}}$ 的差值才有意义。如果模型已经给被选中回答分配了高概率(与参考模型一致),梯度就很小。

### 6.5.5 逐词元 vs 序列级:长度归一化

**一个微妙的问题**:更长的序列天然对数概率更低(求和的项更多,且每项都 $\le 0$)。如果 $|y_w| \gg |y_l|$,损失可能偏向于偏好更短的回答。

**解决方案**:

- **长度归一化 DPO**:把 $\log \pi_{\theta}(y \mid x)$ 替换为 $\frac{1}{|y|}\sum_{t} \log \pi_{\theta}(y_t \mid \cdot)$。一些实现采用了这种做法(SimPO 即采用此法)。
- **标准 DPO**:使用原始求和(不做归一化)。这隐式地惩罚了冗长回答 —— 模型必须对被选中回答中的每个词元都赋予高概率。
- **实际影响**:在基准测试中,长度归一化 DPO 能减少对长度的投机取巧,但可能损害指令遵循质量。生产环境中更常见的是标准(不归一化)版本。

### 6.5.6 标签掩蔽:哪些位置接收梯度

> **DPO 中哪些词元接收梯度**(参见原文 p148 图)

- **提示词元($x$)**:**不接收梯度**。损失仅在回答位置上计算。提示词元提供上下文,但其 logits 不参与 $\log \pi(y \mid x)$ 的计算。
- **被选中回答词元($y_w$)**:**所有词元都接收梯度**。每个 $y_w^t$ 都参与求和。梯度推高它们的概率。
- **被拒绝回答词元($y_l$)**:**所有词元都接收梯度**。每个 $y_l^t$ 都参与求和。梯度压低它们的概率。
- **填充词元(padding tokens)**:**不接收梯度**。通过 attention mask 被掩蔽掉。

### 6.5.7 伪代码:DPO 训练步

```python
# DPO 前向 + 反向(PyTorch 风格)
def dpo_loss(model, ref_model, batch, beta=0.1):
    """One DPO training step."""
    # batch contains: input_ids_chosen, input_ids_rejected,
    #                 labels_chosen, labels_rejected (prompt masked to -100)
    # 1. 前向传播:获取逐词元的对数概率
    logps_chosen = get_sequence_logprob(model, batch["chosen"])
    logps_rejected = get_sequence_logprob(model, batch["rejected"])

    # 2. 参考模型的对数概率(预先计算或在此处计算)
    with torch.no_grad():
        ref_logps_chosen = get_sequence_logprob(ref_model, batch["chosen"])
        ref_logps_rejected = get_sequence_logprob(ref_model, batch["rejected"])

    # 3. 计算隐式奖励间隔
    chosen_rewards = beta * (logps_chosen - ref_logps_chosen)
    rejected_rewards = beta * (logps_rejected - ref_logps_rejected)

    # 4. DPO 损失 = -log(sigmoid(chosen_reward - rejected_reward))
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


def get_sequence_logprob(model, sequences):
    """Sum of log-probs over response tokens only."""
    outputs = model(sequences["input_ids"], attention_mask=sequences["mask"])
    logits = outputs.logits[:, :-1, :]        # 为 next-token 预测做位移
    # 收集真实词元的对数概率
    labels = sequences["labels"][:, 1:]        # 位移后的标签
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # 掩码:仅对回答词元求和(labels != -100)
    mask = (labels != -100).float()
    return (token_logps * mask).sum(dim=-1)    # 形状:[batch_size]
```

### 6.5.8 常见陷阱

> **DPO 实现陷阱**(参见原文 p149 图)

- **忘记掩蔽提示**:如果把提示词元也纳入对数概率求和,模型就在优化提示的似然(毫无用处),而且有效的 $\beta$ 也会出错。
- **用平均替代求和**:$\frac{1}{T}\sum_{t} \log \pi$ 与 $\sum_{t} \log \pi$ 会给出不同的隐式长度惩罚。$\pi_{\theta}$ 与 $\pi_{\text{ref}}$ 之间必须保持一致。
- **陈旧的参考模型**:如果 $\pi_{\text{ref}}$ 与 $\pi_{\theta}$ 相差太远(例如用基座模型而非微调后的模型作参考),KL 项会主导损失、梯度消失。**解决办法**:用 SFT 检查点(而非基座模型)作为参考。
- **$\beta$ 过大**:会放大对数概率之差 → sigmoid 饱和 → 梯度为零。建议从 $\beta = 0.1$ 开始,在 $[0.05, 0.5]$ 范围内调参。
- **$\beta$ 过小**:理论上允许更大程度地偏离参考模型(更弱的 KL 约束),但梯度 $\propto \beta \cdot \sigma(-\beta h)$ 会变得极小 → 损失曲面平坦 → 收敛极慢。模型得到了「可以远离参考」的许可,却几乎得不到告诉它该往哪走的信号。

## 6.6 DPO 的变体及其各自的失效场景

### DPO 何时失效

1. **分布漂移**:偏好数据来自旧模型。当前策略生成的文本与之不同 → 损失是在无关样本上做优化。
2. **缺乏探索**:无法发现数据集中没有出现的行为。模型陷于局部最优。
3. **参考坍缩(reference collapse)**:若参考模型过强,策略无法移动;若过弱,则失去正则化作用。
4. **数据质量**:带噪声的标签会毒害训练。与 PPO 会对大量样本取平均不同,DPO 会逐对记忆偏好对。
5. **偏好数据的多样性**:要确保选中/拒绝对覆盖质量差异的全谱(而不只是「好 vs 极差」)。在**思路**而非仅**质量**上有差异的偏好对,能教会模型更丰富的策略判别力。

## 6.7 $\beta$ 选择指南

| $\beta$ | 状态(regime) | 适用场景 |
|---|---|---|
| 0.01 | 极激进 | 仅当数据极其干净、且需要大幅分布漂移时 |
| 0.05 | 激进 | 数据质量好,希望在 SFT 之上有明显提升 |
| 0.1 | 标准 | 默认起点。质量与稳定性的良好平衡 |
| 0.2 | 保守 | 数据有噪声,或模型已接近期望行为 |
| 0.5 | 极保守 | 安全微调场景,绝不能破坏已有能力 |

## 6.8 DPO 批次大小配置与扩展

与在单序列词元预测上操作的标准 SFT 不同,DPO 使用一个**成对损失(pairwise loss)**,把偏好序列与不偏好序列进行比较。这从根本上改变了显存占用与优化稳定性。

### 6.8.1 全局批次大小目标

跨 DPO 实现的经验证据确立了最优全局批次大小区间

$$
B_{\text{global}} \in [32,\, 128] \quad (6.12)
$$

- $B_{\text{global}} < 32$:隐式奖励估计中出现严重的梯度噪声 → 策略在对齐目标(有用性 vs 安全性)之间出现破坏性的来回震荡。
- $B_{\text{global}} > 128$:收敛速度的边际收益递减;分布式计算中通信开销过高。

### 6.8.2 数学分解

由于 DPO 同时加载两份模型副本(活跃策略 $\pi_{\theta}$ + 冻结参考 $\pi_{\text{ref}}$),单序列的显存占用翻倍。全局批次大小分解为

$$
B_{\text{global}} = B_{\text{micro}} \times N_{\text{GPUs}} \times K_{\text{accum}} \quad (6.13)
$$

- $B_{\text{micro}}$:单设备微批次大小(每次前向传播处理的偏好对数)。
- $N_{\text{GPUs}}$:并行处理数据的设备数。
- $K_{\text{accum}}$:权重更新前的梯度累积步数。

**配对倍数**:单个 DPO 数据实例包含一个提示($x$)、一个选中回答($y_w$)和一个拒绝回答($y_l$)。每个微批次的实际张量负载为

$$
T_{\text{sequences}} = 2 \times B_{\text{micro}} \quad (6.14)
$$

对于在 80GB GPU、上下文长度 4096–8192 词元下运行的 >7B 参数模型,物理上限被严格约束在 $B_{\text{micro}} \in [1,\, 2]$。

### 6.8.3 分布式扩展配置

表 6.1:DPO 训练的分布式扩展档位(目标 $B_{\text{global}} = 64$)。

| 配置 | 单 GPU | 8-GPU 节点 |
|---|---|---|
| $B_{\text{global}}$ | 64 | 64 |
| $B_{\text{micro}}$ | 2(4 条序列) | 2(4 条序列) |
| $N_{\text{GPUs}}$ | 1 | 8 |
| $K_{\text{accum}}$ | 32 步 | 4 步 |
| 吞吐量 | 串行/较慢 | 高并行吞吐 |

### 6.8.4 VRAM 优化:预计算参考对数概率

DPO 损失为

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,\, y_w,\, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right] \quad (6.15)
$$

由于 $\pi_{\text{ref}}$ 在整个训练过程中完全静态,它的输出可以**预先计算**:

> **参考模型逐出(eviction)策略**(参见原文 p151 图)

1. 在训练开始之前,仅用 $\pi_{\text{ref}}$ 对数据集 $\mathcal{D}$ 执行一次前向传播。
2. 将标量 $\log \pi_{\text{ref}}(y_w \mid x)$ 和 $\log \pi_{\text{ref}}(y_l \mid x)$ 缓存到磁盘。
3. 把 $\pi_{\text{ref}}$ 完全从 GPU 显存中逐出。

**结果**:可用 GPU 显存翻倍 → 可把 $B_{\text{micro}}$ 从 1–2 提升到 4–8,最大化硬件利用率与训练吞吐。

**实现**:在 TRL 中,于 `DPOConfig` 中设置 `precompute_ref_log_probs=True`。对于 70B 模型,这可在整个集群上节省约 140GB 的 GPU 显存。

## 6.9 DPO 的扩展与变体

直接偏好优化(DPO)通过推导奖励函数与最优策略之间的闭式映射,把 RLHF 重新表述为一个监督学习问题。标准 DPO 损失为

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(q,\, y_w,\, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w \mid q)}{\pi_{\text{ref}}(y_w \mid q)} - \beta \log \frac{\pi_{\theta}(y_l \mid q)}{\pi_{\text{ref}}(y_l \mid q)} \right) \right],
$$

其中 $y_w$ 是偏好(获胜)回答,$y_l$ 是不偏好(落败)回答,$\beta$ 控制 KL 惩罚的强度。下面各小节介绍最重要的扩展与变体。

### 6.9.1 f-DPO —— 广义 f-散度 DPO

> **超越反向 KL**(参见原文 p152 图)

标准 DPO 使用**反向 KL 散度(reverse KL divergence)**作为策略与参考之间的正则化器。反向 KL 是**寻峰(mode-seeking)**的:它倾向于把概率质量集中在少数高奖励回答上。**前向 KL(forward KL)**则是**覆盖(mode-covering)**的:它把概率质量摊开,覆盖所有合理回答。f-DPO [177] 将其推广到任意 f-散度(f-divergence),让从业者可以在这些行为之间权衡。

f-DPO 损失用 f-散度生成元的导数替换了对数比

$$
\mathcal{L}_{\text{f-DPO}} = -\,\mathbb{E} \left[ f'\!\left( \frac{\pi_{\theta}(y_w \mid q)}{\pi_{\text{ref}}(y_w \mid q)} \right) - f'\!\left( \frac{\pi_{\theta}(y_l \mid q)}{\pi_{\text{ref}}(y_l \mid q)} \right) \right],
$$

其中 $f'$ 是 f-散度生成元函数的导数。

**TRL 中的 f-散度选项**:

- `reverse_kl`:$f'(u) = \log u$。标准 DPO,寻峰。
- `forward_kl`:$f'(u) = -1/u$。覆盖,更好的多样性。
- `js_divergence`:$f'(u) = \log\big(2u/(u + 1)\big)$。寻峰/覆盖的平衡。
- `alpha_divergence`:$f'(u) = u^{\alpha - 1}$。在前向与反向 KL 之间插值。

```python
from trl import DPOConfig, DPOTrainer

# Jensen-Shannon 散度(平衡)
config = DPOConfig(
    f_divergence_type="js_divergence",
    beta=0.1,
)

# Alpha 散度(alpha=0:前向 KL,alpha=1:反向 KL)
config_alpha = DPOConfig(
    f_divergence_type="alpha_divergence",
    f_alpha_divergence_coef=0.5,   # alpha 参数
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
)
```

**何时使用 f-DPO**:

- 希望在多样性与质量之间取得平衡时,使用 JS 散度。
- 多样性至上的创意任务,使用前向 KL。
- 存在唯一正确答案的任务,使用反向 KL(标准 DPO)。
- 使用 alpha 散度连续插值并调谐权衡。

### 6.9.2 Robust DPO —— 鲁棒 DPO

> **偏好数据中的噪声标签**(参见原文 p153 图)

人类偏好标注是有噪声的:标注者之间会意见不一、会犯错,有时会翻转偏好/不偏好的标签。标准 DPO 把所有标签都视为真值,这可能导致模型对噪声过拟合。Robust DPO [178] 在已知噪声模型下对损失进行了解析去偏。

假设每个标签以概率 $\epsilon$(噪声率)被翻转,则去偏损失为

$$
\mathcal{L}_{\text{robust}} = \frac{(1 - \epsilon)\, \mathcal{L}_{\text{DPO}}(y_w, y_l) - \epsilon\, \mathcal{L}_{\text{DPO}}(y_l, y_w)}{1 - 2\epsilon},
$$

其中 $\mathcal{L}_{\text{DPO}}(y_w, y_l)$ 是把 $y_w$ 视为偏好的标准 DPO 损失,$\mathcal{L}_{\text{DPO}}(y_l, y_w)$ 是标签翻转后的损失。这一校正消除了标签噪声引入的偏差。

**Robust DPO 的直觉**:该公式是一个线性组合,「减去」了翻转标签的贡献。当 $\epsilon = 0$ 时退化为标准 DPO;当 $\epsilon = 0.5$ 时分母趋于零 —— 标签纯属噪声,无法学习。实践中,$\epsilon \in [0.05,\, 0.2]$ 覆盖了大多数真实标注噪声水平。

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="robust",
    label_smoothing=0.1,   # 对应 epsilon = 0.1
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
)
```

### 6.9.3 TR-DPO —— 信赖域 DPO(Trust Region DPO)

> **陈旧参考模型问题**(参见原文 p153 图)

标准 DPO 在整个训练过程中使用固定的参考模型 $\pi_{\text{ref}}$。随着策略 $\pi_{\theta}$ 改进,KL 惩罚 $\beta \log(\pi_{\theta}/\pi_{\text{ref}})$ 不断增长,最终主导损失、阻碍进一步提升。TR-DPO [179] 周期性地更新参考模型,使其追踪当前策略。

TR-DPO 使用**指数移动平均(exponential moving average, EMA)**来更新参考模型

$$
\pi_{\text{ref}}^{(t+1)} \leftarrow \alpha \cdot \pi_{\theta}^{(t)} + (1 - \alpha) \cdot \pi_{\text{ref}}^{(t)},
$$

其中 $\alpha \in (0, 1)$ 是混合系数,每 $T_{\text{sync}}$ 个梯度步执行一次。

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="sigmoid",                  # 标准 DPO 损失
    sync_ref_model=True,                  # 启用 TR-DPO
    ref_model_mixup_alpha=0.6,            # alpha:掺入当前策略的比例
    ref_model_sync_steps=512,             # T_sync:每 512 步同步一次
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
)
```

**何时使用 TR-DPO**:

- 长时间训练运行中,策略已经漂移得远离初始参考模型。
- 当观察到 DPO 损失因 KL 惩罚主导而过早进入平台期时。
- 迭代式 DPO 流水线,会从当前策略采集新的偏好数据。
- $\alpha$ 接近 1 表示参考更新更快;接近 0 表示更新更慢。

### 6.9.4 EXO —— 精确优化(Exact Optimisation)

> **DPO 的 KL 方向问题**(参见原文 p154 图)

DPO 是在反向 KL 约束下求解最优策略推导出来的。然而,得到的损失在奖励空间中实际优化的却是一个**前向 KL 目标**,这是错误的方向。EXO [180] 通过使用**反向 KL 概率匹配**来纠正这一点 —— 这才是对齐理论上正确的目标。

EXO 最小化模型分布与目标(奖励最优)分布之间的反向 KL

$$
\mathcal{L}_{\text{EXO}} = \mathbb{E}_{y \sim \pi_{\theta}} \left[ \log \frac{\pi_{\theta}(y \mid q)}{p^{*}(y \mid q)} \right],
$$

其中 $p^{*}(y \mid q) \propto \pi_{\text{ref}}(y \mid q)\, \exp\big(r(y, q)/\beta\big)$ 是最优策略。实践中,EXO 利用可得的偏好对来近似它

$$
\mathcal{L}_{\text{EXO}} \approx -\,\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_{\text{ref}}(y_w \mid q)}{\pi_{\theta}(y_w \mid q)} - \beta \log \frac{\pi_{\text{ref}}(y_l \mid q)}{\pi_{\theta}(y_l \mid q)} \right) \right].
$$

注意,与 DPO 相比,$\pi_{\theta}$ 与 $\pi_{\text{ref}}$ 的角色互换了。

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="exo_pair",
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
)
```

### 6.9.5 NCA —— 噪声对比对齐(Noise Contrastive Alignment)

> **DPO 的似然坍缩**(参见原文 p155 图)

DPO 已知的一种失效模式是**似然坍缩(likelihood collapse)**:模型学会降低落败回答的概率,但同时也降低了获胜回答的概率(因为损失只关心两者之差)。NCA [181] 加入一个**绝对似然项**来阻止这种情况。

NCA 把对齐重新表述为**噪声对比估计(noise-contrastive estimation)**。其损失包含三项

$$
\mathcal{L}_{\text{NCA}} = -\log \sigma(r_w) - \tfrac{1}{2} \log \sigma(-r_w) - \tfrac{1}{2} \log \sigma(-r_l),
$$

其中 $r_y = \beta \log\big(\pi_{\theta}(y \mid q)/\pi_{\text{ref}}(y \mid q)\big)$ 是隐式奖励。第一项鼓励 $y_w$ 获得高奖励;第二、三项则惩罚 $y_w$ 与 $y_l$ 二者都获得高奖励(从而防止坍缩)。

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="nca_pair",
    beta=0.01,             # 较小的 beta:绝对似然项占主导
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
)
```

**何时使用 NCA**:

- 当观察到 DPO 训练中获胜回答的概率在下降时。
- 对于绝对回答质量(而非仅相对排序)也很重要的任务。
- 使用较小的 $\beta$(如 0.01),给绝对似然项更大的权重。

### 6.9.6 SLiC-HF —— 序列似然校准(Sequence Likelihood Calibration)

> **作为更简单替代方案的合页损失**(参见原文 p155 图)

DPO 中的 log-sigmoid 损失是平滑的,但当间隔较大时收敛可能很慢。SLiC-HF [182] 使用**合页损失(hinge loss)**:当间隔超过某个阈值时损失为零,否则线性增长。这更简单、更快,而且出奇地有竞争力。

SLiC-HF 损失为

$$
\mathcal{L}_{\text{SLiC}} = \max\!\left( 0,\; \delta - \beta \log \frac{\pi_{\theta}(y_w \mid q)}{\pi_{\text{ref}}(y_w \mid q)} + \beta \log \frac{\pi_{\theta}(y_l \mid q)}{\pi_{\text{ref}}(y_l \mid q)} \right),
$$

其中 $\delta$ 是间隔阈值。当模型已经在获胜与落败回答之间给出 $\delta$ 的间隔时,损失为零。

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="hinge",
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
)
```

### 6.9.7 Iterative RPO —— 推理偏好优化(Reasoning Preference Optimisation)

> **DPO 遗忘了如何生成**(参见原文 p156 图)

标准 DPO 训练模型去区分获胜与落败回答。但对于推理任务,模型还需要**生成**正确的推理过程。一个只会判别而不会生成的模型,在推理时毫无用处。RPO 在获胜回答上加入一个 NLL(负对数似然,negative log-likelihood)项,以确保模型学会生成它。

RPO 损失把 DPO 与 SFT 结合起来

$$
\mathcal{L}_{\text{RPO}} = \lambda_{1}\, \mathcal{L}_{\text{DPO}}(y_w, y_l) + \lambda_{2}\, \mathcal{L}_{\text{NLL}}(y_w),
$$

其中 $\mathcal{L}_{\text{NLL}}(y_w) = -\log \pi_{\theta}(y_w \mid q)$ 是在获胜回答上的标准语言建模损失。

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="sigmoid",        # 标准 DPO 损失
    rpo_alpha=1.0,              # NLL 正则化权重(RPO)
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
)
```

**何时使用 RPO**:

- 推理任务(数学、代码、逻辑),模型必须生成逐步求解过程。
- 当 DPO 训练导致模型丧失流畅性或生成质量时。
- 迭代式流水线:生成 rollout → 标注 → 用 RPO 训练 → 重复。
- NLL 项起正则化器的作用,防止策略坍缩。

### 6.9.8 SimPO —— 简单偏好优化(Simple Preference Optimisation)

> **无参考模型的偏好学习**(参见原文 p156 图)

DPO 需要一个参考模型来计算隐式奖励。这会使显存占用翻倍,并增加复杂度。SimPO [183] 用回答的**平均对数概率**作为隐式奖励,并引入一个**长度归一化项**以防止模型偏好短回答,从而**消除了参考模型**。

SimPO 把隐式奖励定义为

$$
r_{\text{SimPO}}(y \mid q) = \frac{\beta}{|y|} \log \pi_{\theta}(y \mid q),
$$

损失为

$$
\mathcal{L}_{\text{SimPO}} = -\,\mathbb{E} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_{\theta}(y_w \mid q) - \frac{\beta}{|y_l|} \log \pi_{\theta}(y_l \mid q) - \gamma \right) \right],
$$

其中 $\gamma > 0$ 是**目标奖励间隔**,确保获胜回答的奖励严格高于落败回答至少 $\gamma$。

**SimPO vs DPO vs ORPO**:

- **DPO**:使用参考模型;基于比值的隐式奖励。
- **ORPO**:无参考模型;在 SFT 损失上加入赔率比(odds-ratio)项。
- **SimPO**:无参考模型;长度归一化的对数概率奖励 + 间隔。
- SimPO 比 DPO 更简单(无需参考模型),且比 ORPO 更有原则性。
- SimPO 中的长度归一化至关重要:没有它,模型会偏好长回答。

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    loss_type="simpo",
    simpo_gamma=0.5,        # 目标奖励间隔 gamma
    beta=2.5,               # 长度归一化系数
    # 无需 ref_model!
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,         # SimPO 无需参考模型
    args=config,
    train_dataset=dataset,
)
```
