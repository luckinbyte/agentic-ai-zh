# 第 5 章 PPO —— 近端策略优化(Proximal Policy Optimization)

## 5.1 动机与历史

**问题**:原始(vanilla)策略梯度更新对步长没有任何约束。一个不走运的批次(batch)就可能把策略推入一个会生成垃圾输出的区域 → 垃圾输出得到低奖励 → 下一个梯度让情况更糟 → 不可恢复的崩溃(collapse)。

**解决方案演进史**:

1. TRPO [167](2015):约束新旧策略之间的 KL 散度。效果完美,但需要昂贵的二阶优化(Fisher 信息矩阵、共轭梯度)。
2. PPO(2017)[168]:用一个简单的一阶截断目标函数(clipped objective)达到类似的稳定性。实现简单 10 倍,效果几乎一样好,并且可以轻松扩展到分布式训练。

## 5.2 截断目标函数

PPO 的核心创新是一个截断的替代目标函数(clipped surrogate objective),它能在保持实现简单的同时,防止破坏性的过大策略更新。

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\hat{A}_t \right) \right] \quad (5.1)
$$

其中概率比为 $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$。

### 截断的直觉 —— 关键洞见

`min` 算子构造了一个悲观的下界:

- 好动作($\hat{A} > 0$):我们希望提高它的概率。替代项 $r\hat{A}$ 随 $r$ 增大而增大,但 clip 把收益上限封在 $r = 1 + \epsilon$。即「不要因为一个好的样本就贪心。」
- 坏动作($\hat{A} < 0$):我们希望降低它的概率。$r\hat{A}$ 随 $r$ 减小而改善,但 clip 把收益上限封在 $r = 1 - \epsilon$。即「不要因为一个坏的样本就遗忘得太激进。」

净效果:每次更新策略最多改变 $\pm 20\%$。既防止灾难性崩溃,也防止过度自信的特化(overconfident specialization)。

## 5.3 完整的 PPO 损失

$$
\mathcal{L} = L^{\text{CLIP}} - c_1 \underbrace{\left( V_\theta(s_t) - V_t^{\text{target}} \right)^2}_{\text{价值损失}} + c_2 \underbrace{H[\pi_\theta(\cdot \mid s_t)]}_{\text{熵奖励}} \quad (5.2)
$$

- **价值损失(value loss)**($c_1 = 0.1$):训练 critic(评论网络)预测回报(returns)。同样会做截断以保持稳定性。
- **熵奖励(entropy bonus)**($c_2 = 0.01$):防止过早收敛到确定性策略。对探索(exploration)至关重要。

## 5.4 PPO 梯度与更新规则的推导

本节追溯从 RL 目标到 PPO 更新规则的数学路径,说明截断替代为何有效。

### 5.4.1 第 1 步:RL 目标

目标是在策略下最大化期望累计奖励:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} r_t \right] \quad (5.3)
$$

### 5.4.2 第 2 步:策略梯度定理

$J(\theta)$ 对策略参数的梯度为:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t \right] \quad (5.4)
$$

其中 $\hat{A}_t$ 是优势函数(advantage function),表示动作 $a_t$ 相比状态 $s_t$ 下平均动作好多少。这里用优势替代完整回报,以降低方差。

### 5.4.3 第 3 步:面向异策略数据的重要性采样

PPO 用 $\pi_{\theta_{old}}$ 采集数据,却去更新 $\pi_\theta$。为了修正这种分布不匹配,采用重要性采样(importance sampling):

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t \right] \quad (5.5)
$$

定义概率比 $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$。利用恒等式 $\nabla_\theta \log f = \dfrac{\nabla_\theta f}{f}$,可得:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{old}}} \left[ \nabla_\theta r_t(\theta) \cdot \hat{A}_t \right] \quad (5.6)
$$

这意味着我们要最大化如下替代目标:

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \cdot \hat{A}_t \right] \quad (5.7)
$$

### 5.4.4 第 4 步:无约束替代的问题

$L^{\text{CPI}}$ 是一个有效的目标,但在没有约束的情况下,单个梯度步就可能把 $r_t(\theta)$ 推得远离 1.0,导致:

- 重要性权重变得极端 → 高方差
- 策略进入未经测试的区域 → 奖励模型(reward model)给出不可靠的分数
- 灾难性崩溃:策略生成垃圾输出,无法恢复

**TRPO 的解法**:约束 $D_{KL}(\pi_{\theta_{old}} \,\|\, \pi_\theta) \le \delta$。需要二阶方法(昂贵)。

### 5.4.5 第 5 步:PPO 的截断替代(一阶近似)

PPO 用一个截断目标函数替代硬 KL 约束,仅用一阶梯度就能达到类似行为:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\hat{A}_t \right) \right] \quad (5.8)
$$

**梯度的推导**:

令 $L_t = \min(r_t\hat{A}_t,\, \bar{r}_t\hat{A}_t)$,其中 $\bar{r}_t = \text{clip}(r_t,\, 1-\epsilon,\, 1+\epsilon)$。

$$
\nabla_\theta L_t =
\begin{cases}
\nabla_\theta r_t(\theta) \cdot \hat{A}_t & \text{若 } r_t\hat{A}_t < \bar{r}_t\hat{A}_t \text{(未截断项更小)} \\
0 & \text{若 } r_t\hat{A}_t \ge \bar{r}_t\hat{A}_t \text{(截断项更小,梯度 } = 0\text{)}
\end{cases} \quad (5.9)
$$

展开各条件:

- 当 $\hat{A}_t > 0$ 且 $r_t < 1 + \epsilon$:梯度正常流动 —— 鼓励策略提高 $\pi_\theta(a_t \mid s_t)$。
- 当 $\hat{A}_t > 0$ 且 $r_t \ge 1 + \epsilon$:梯度为零 —— 策略已提升得足够多,停止推动。
- 当 $\hat{A}_t < 0$ 且 $r_t > 1 - \epsilon$:梯度正常流动 —— 鼓励策略降低 $\pi_\theta(a_t \mid s_t)$。
- 当 $\hat{A}_t < 0$ 且 $r_t \le 1 - \epsilon$:梯度为零 —— 策略已降低得足够多,停止推动。

### 5.4.6 第 6 步:完整的 PPO 更新规则

把截断策略损失、价值损失与熵奖励组合起来:

$$
\theta_{k+1} = \theta_k + \alpha \cdot \nabla_\theta \left[ L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 H[\pi_\theta] \right] \quad (5.10)
$$

其中:

$$
L^{\text{VF}}(\theta) = \left( V_\theta(s_t) - V_t^{\text{target}} \right)^2 \quad \text{(价值函数回归损失)} \quad (5.11)
$$

$$
H[\pi_\theta] = -\sum_{a} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t) \quad \text{(策略的熵)} \quad (5.12)
$$

### 小结:为什么这样做有效

1. 策略梯度定理给出了改进策略的方向。
2. 重要性采样让我们可以跨多个 epoch 复用来自 $\pi_{\theta_{old}}$ 的数据。
3. 截断防止重要性权重变得极端,从而保持更新安全。
4. `min` 算子确保我们始终取(截断、未截断)两者中更保守的那个 —— 一个关于改进的悲观下界。
5. 结果:仅用一阶梯度即可(以概率 1)实现单调改进。无需 Hessian 矩阵、无需共轭梯度、无需线搜索。

## 5.5 Rollout Buffer 与 Rollout

在 PPO 中,数据管理依赖一种专门的短期存储系统,称为 Rollout Buffer(采样缓冲区)。与那些在 replay buffer(经验回放缓冲区)中无限期存储经验的异策略(off-policy)算法(如 DQN)不同,PPO 需要一种临时性结构来满足其同策略(on-policy)的数学约束。

### 5.5.1 什么是 Rollout?

一次 rollout(轨迹,trajectory)是智能体在环境中运行其当前策略所生成的一段交互序列:

- **过程**:智能体观察一个状态,选择一个动作,获得奖励,并转移到下一个状态。如此重复固定的步数,或直到回合(episode)结束。
- **在 LLM/RLHF 中**:一次 rollout 包含从数据集中取出一个 prompt(提示),让语言模型逐词元(token)生成一个完整序列,直到遇到文本结束标记。每个 token 就是一个「步骤」。

### 5.5.2 Rollout Buffer

Rollout buffer 临时存储在 rollout 阶段收集的所有数据。对每个生成的 token/步骤,它记录:

$$
\mathcal{B} = \left\{ \left( s_t,\, a_t,\, \log \pi_{\theta_{old}}(a_t \mid s_t),\, r_t,\, V(s_t) \right) \right\}_{t=1}^{T} \quad (5.13)
$$

- $s_t, a_t, r_t$:第 $t$ 步的状态、所采取的动作、奖励。
- $\log \pi_{\theta_{old}}(a_t \mid s_t)$:在生成该动作的精确策略下采取该动作的对数概率(用于计算比率)。
- $V(s_t)$:价值函数的基线预测(用于计算 GAE 优势)。

### 5.5.3 Rollout Buffer 的生命周期

该缓冲区按严格的三阶段时钟周期运行:

1. **收集(Collect)**:当前策略与环境交互,用新鲜轨迹填满缓冲区(对一个 batch=128、max_tokens=512 的 70B 模型而言:每次 rollout 多达 65K 个 token 级别的转移)。
2. **训练(Train)**:沿轨迹计算 GAE 优势。运行 $K$ 个 epoch(通常 3–10)的小批量(mini-batch)梯度下降,用截断目标函数更新策略权重。
3. **清空(Purge)**:整个缓冲区被完全清空。因为 PPO 是同策略的,旧策略生成的数据无法安全地用于下一个更新周期 —— 比率 $r_t(\theta)$ 会变得过时,截断保证也随之失效。

### Rollout Buffer 与 Replay Buffer 的对比

- **Replay Buffer(DQN、SAC)**:异策略。无限期存储上百万条转移。随机采样。数据跨多次更新复用。
- **Rollout Buffer(PPO、GRPO)**:同策略。存储一个批次的轨迹。使用少数几个 epoch 后即整体丢弃。每个周期都需要新鲜数据。

这就是为什么 PPO 需要持续不断地生成 —— 缓冲区在每次更新后都被清空,从而要求新鲜的 rollout。这让生成瓶颈(占 wall-clock 时间的 60–70%)尤为痛苦。

### RLHF 场景中的 vLLM

在 RLHF 训练中,vLLM 用于生成阶段(占 wall-clock 时间的 60–70%)。策略模型生成 rollout,再由奖励模型打分。关键收益:

- **批量生成(Batched generation)**:跨多个 prompt 并行生成 256 条以上响应。
- **内存效率**:容纳更多并发生成 → 在生成瓶颈期间获得更高的 GPU 利用率。
- **前缀共享(Prefix sharing)**:当为每个 prompt 生成 $N = 8$ 条响应(GRPO)时,prompt 的 KV 只计算一次,并在全部 8 条之间共享 —— 无冗余 prefill。
- **集成**:OpenRLHF、TRL 等框架把 vLLM 用作生成后端,将生成工作进程(vLLM)与训练工作进程(DeepSpeed/FSDP)分离。

## 5.6 PPO 用于 RLHF:完整循环

### 一个 70B 聊天模型的具体 PPO 步骤

**设置**:128 条 prompt 的批次,Llama-3-70B 策略,最长 512 个 token。

- **第 1 步 —— 生成**:采样 128 条响应(temperature=0.7、top-p=0.9)。耗时占总时间的 60%。
- **第 2 步 —— 打分**:奖励模型对每个(prompt, response)对打分。范围:0.2–0.95。
- **第 3 步 —— KL**:计算逐 token 的 KL:$\text{KL}_t = \log \pi_\theta(y_t \mid y_{<t}) - \log \pi_{\text{ref}}(y_t \mid y_{<t})$。token 上的平均 KL 通常为 3–8。
- **第 4 步 —— 最终奖励**:$R = r_{\text{RM}} - 0.05 \times \text{mean\_KL}$(仅在最后一个 token 处)。
- **第 5 步 —— GAE**:用价值头(value head)预测为每个 token 位置计算 $\hat{A}_t$。对优势做白化(whiten,零均值、单位方差)。
- **第 6 步 —— 更新**:在大小为 16 的小批量上做 4 个 epoch 的 SGD。截断比率 $\epsilon = 0.2$。梯度范数裁剪在 1.0。

**结果**:每步策略胜率提升约 0.005。10K 步后:相对 SFT 绝对提升 5–10%。

### LLM 强化学习中的分词陷阱

在计算逐 token 的 KL 惩罚和优势时,请记住:**分词(tokenization)决定了什么是一个「步骤」**。一个概念上的单一动作(例如输出「2024」)根据分词器不同可能跨越 1–4 个 token。这带来一些微妙的问题:

- **KL 核算**:同样的语义内容以不同方式分词时,逐 token 的 KL 之和会得到不同的总量(例如,被拆成更多子词的罕见词累计的 KL 惩罚更高)。
- **信用分配(credit assignment)**:GAE 按 token 位置分配优势 —— 但语义上的「决策」往往跨越多个 token。模型真正「决策」之处通常只在一个词的首个 token;后续的子词 token 在很大程度上是确定性的。
- **奖励放置**:仅在最后一个 token 放置奖励,意味着前面所有 token 都必须通过 GAE 向后传播信用 —— 更长的响应会遭受更稀释的信号。

**缓解**:一些系统按序列长度归一化 KL,采用词级奖励塑形(reward shaping),或在语义边界处施加奖励,而非最后一个 token。

## 5.7 详细机制:Logits 与策略更新

PPO 在内存中管理两种截然不同的参数状态,它们共享相同的神经网络拓扑,但在优化过程中持有不同的权重值:

![图 5.1:PPO 端到端流程 —— 从 prompt 批次出发,经生成、奖励打分、KL 计算、优势估计,到截断的策略更新。反馈回路表明更新后的策略被用于下一次生成步骤。](images/part-ii-rl-methods-for-llms/ppo-proximal-policy-optimization/ppo-proximal-policy-optimization-p140-01.png)

### 核心架构:两个网络

1. **策略网络($\pi_\theta$)**:活跃的、在线的网络,由权重 $\theta$ 参数化。在优化过程中通过反向传播持续更新。
2. **旧策略网络($\pi_{\theta_{old}}$)**:一个冻结的快照,由权重 $\theta_{old}$ 参数化。在单个优化周期内充当静态锚点,防止策略漂移得过于剧烈。

### 5.7.1 阶段 1:Rollout(数据收集)

在数据收集期间,智能体与环境交互 $T$ 步。在每个时间步 $t$:

1. 环境给出当前状态/观测 $s_t$(对 LLM 而言:prompt + 已生成的 token)。
2. 状态 $s_t$ 经过当前网络快照($\theta_{old}$)的前向传播。
3. 网络输出原始的、未归一化的值 —— logits $z_{old}$ —— 一个大小为 $|V|$ 的向量(词表大小 32K–128K)。
4. 通过 Softmax 计算概率:

$$
P(a \mid s_t) = \text{Softmax}(z_{old}) = \frac{\exp(z_{old,a})}{\sum_{j=1}^{|V|} \exp(z_{old,j})} \quad (5.14)
$$

5. 从 $P(a \mid s_t)$ 中采样一个动作 $a_t$(下一个 token),并将转移元组 $\langle s_t, a_t, r_t, s_{t+1} \rangle$ 连同 $\log \pi_{\theta_{old}}(a_t \mid s_t)$ 一并存入 rollout buffer。

#### 为什么要存对数概率?

在 rollout 时把 $\log \pi_{\theta_{old}}(a_t \mid s_t)$ 存为标量,可避免在优化阶段重新运行冻结的网络。这为每个小批量省下一次完整的前向传播 —— 对 70B 模型而言意义重大。

### 5.7.2 阶段 2:优化循环(小批量更新)

当 rollout buffer 装满后,PPO 在小批量上运行 $K$ 个 epoch(通常 3–10)。对每个梯度步,利用存储的状态 $s_t$ 为两个策略分别生成 logits。

**旧策略求值(冻结)**:

$$
z_{old} = f(s_t;\, \theta_{old}) \;\longrightarrow\; \log \pi_{\theta_{old}}(a_t \mid s_t) = \text{LogSoftmax}(z_{old})[a_t] \quad (5.15)
$$

*实现捷径:直接复用 rollout 时存储的标量,而非重新计算。*

**活跃策略求值(更新中)**:

$$
z_{new} = f(s_t;\, \theta) \;\longrightarrow\; \log \pi_\theta(a_t \mid s_t) = \text{LogSoftmax}(z_{new})[a_t] \quad (5.16)
$$

由于 $\theta$ 在每个小批量梯度步后都会更新,$z_{new}$ 在整个优化循环中持续变化,而 $z_{old}$ 则保持完全静态。

### 5.7.3 从 Logits 到概率比

PPO 的核心比率衡量一个动作在新策略下相比旧策略的可能性增减程度:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)} \quad (5.17)
$$

为避免原始概率相除导致灾难性的数值下溢/上溢,该计算在对数空间进行:

$$
\log \pi_\theta(a_t \mid s_t) = \text{LogSoftmax}(z_{new})[a_t] \quad (5.18)
$$

$$
\log \pi_{\theta_{old}}(a_t \mid s_t) = \text{LogSoftmax}(z_{old})[a_t] \quad (5.19)
$$

通过对其差值取指数恢复比率:

$$
r_t(\theta) = \exp\!\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{old}}(a_t \mid s_t) \right) \quad (5.20)
$$

该比率被注入 PPO 截断目标函数:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\hat{A}_t \right) \right] \quad (5.21)
$$

#### 截断如何起作用

- 若 $\hat{A}_t > 0$(好动作):比率被截断在 $1 + \epsilon$ —— 不能过度利用好动作。
- 若 $\hat{A}_t < 0$(坏动作):比率被截断在 $1 - \epsilon$ —— 不能过度惩罚坏动作。
- `min(·)` 确保我们始终取更保守的估计。

**结果**:在信任域(trust region)内实现单调改进 —— 无灾难性崩溃。

### 5.7.4 PPO 权重生命周期

表 5.1:$\theta$ 与 $\theta_{old}$ 在 PPO 各训练阶段中的演化。

| 阶段 | 活跃 $\theta$ | 旧 $\theta_{old}$ | 比率 $r_t(\theta)$ |
|---|---|---|---|
| 1. Rollout 开始 | 活跃副本 | 同一活跃副本 | 恒为 1.0(恒等) |
| 2. 批次第 1 步 | 计算梯度 | 冻结 | 1.0(初始步) |
| 3. 批次第 $N$ 步 | 修改中($\theta \ne \theta_{old}$) | 冻结 | 偏离 1.0(如 1.06、0.94) |
| 4. 截断激活 | 受 $\epsilon$ 约束 | 冻结 | 困在边界处($1 \pm \epsilon$) |
| 5. 优化结束 | 高度优化 | 丢弃 | 不适用 |
| 6. 下一周期 | $\theta \rightarrow \theta_{old}$ | 接收新鲜 $\theta$ | 重置回 1.0 |

### 5.7.5 连续动作空间扩展

对于连续动作空间(对 LLM 不典型,但对机器人 RL 很重要),网络输出分布参数,而非离散 logits:

- 预测的均值向量 $\mu$

![图 5.2:连续动作空间下策略网络输出分布参数(均值 $\mu$ 与标准差 $\sigma$),并通过高斯对数概率密度函数计算 log 概率,再代入与离散情形相同的截断目标函数。](images/part-ii-rl-methods-for-llms/ppo-proximal-policy-optimization/ppo-proximal-policy-optimization-p141-02.png)

- 预测的标准差向量 $\sigma$

通过对数高斯 PDF 计算 log 概率:

$$
\log \pi(a_t \mid s_t) = -\frac{1}{2} \left( \frac{a_t - \mu}{\sigma} \right)^2 - \log(\sigma) - \frac{1}{2} \log(2\pi) \quad (5.22)
$$

随后比率 $r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{old}})$ 以完全相同的方式计算,并送入同一个截断目标函数。

## 5.8 TRL 实现

HuggingFace 的 TRL 库 [176] 为 LLM 提供了所有主流 RL 方法的产品级实现。

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from peft import LoraConfig

# Model setup  模型设置
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto",
    peft_config=LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj", "v_proj",
                                                                 "k_proj", "o_proj"])
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# PPO config with all critical hyperparameters  含全部关键超参数的 PPO 配置
ppo_config = PPOConfig(
    learning_rate=1.5e-6,        # Low LR for stability  低学习率以保证稳定
    batch_size=128,              # Prompts per step  每步的 prompt 数
    mini_batch_size=16,          # Gradient accumulation unit  梯度累积单元
    ppo_epochs=4,                # Epochs per batch (reuse data)  每批次的 epoch 数(复用数据)
    gamma=1.0,                   # No discounting (single turn)  无折扣(单轮)
    lam=0.95,                    # GAE lambda  GAE 的 lambda
    cliprange=0.2,               # PPO epsilon  PPO 截断比率 epsilon
    cliprange_value=0.2,         # Value function clipping  价值函数截断
    vf_coef=0.1,                 # Value loss coefficient  价值损失系数
    init_kl_coef=0.05,           # Initial KL penalty  初始 KL 惩罚系数
    target_kl=6.0,               # Adaptive KL target  自适应 KL 目标
    whiten_rewards=True,         # Normalize advantages  归一化优势
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
)
ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer,
                         dataset=prompt_dataset, data_collator=collator)

# Training loop  训练循环
for batch in ppo_trainer.dataloader:
    # 1. Generate responses  生成响应
    query_tensors = batch["input_ids"]
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=512,
        temperature=0.7, top_p=0.9, do_sample=True
    )

    # 2. Score with reward model  用奖励模型打分
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
    rewards = [torch.tensor(reward_model.score(q, r)) for q, r in zip(batch["query"], texts)]

    # 3. PPO update (handles KL, GAE, clipping internally)  PPO 更新(内部处理 KL、GAE、截断)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    # Monitor: stats["ppo/mean_scores"], stats["ppo/policy/approx_kl"]  监控指标
```

## 5.9 关键超参数

| 参数 | 典型值 | 调错的后果 |
|---|---|---|
| `cliprange` | 0.2 | 太低:学不动。太高:不稳定。 |
| `init_kl_coef` | 0.01–0.1 | 太低:奖励黑客(reward hacking)。太高:卡在 SFT。 |
| `target_kl` | 4–8 | 自适应控制器的目标。越低越保守。 |
| `ppo_epochs` | 4 | 太多:对批次过拟合。太少:浪费生成算力。 |
| `learning_rate` | $1\text{–}5 \times 10^{-6}$ | 太高:灾难性遗忘。 |
| `batch_size` | 64–256 | 更大:梯度更平滑,但生成算力更多。 |
| `temperature` | 0.7–1.0 | 更低:探索更少。更高:优势更嘈杂。 |
