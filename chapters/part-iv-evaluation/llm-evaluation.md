# 第 14 章 LLM 评估(LLM Evaluation)

评估(evaluation)是任何严谨机器学习流水线的基石,然而它或许是大型语言模型开发中最被低估的环节。在经典的监督学习中,带有真值标签的留出测试集能提供干净的信号;而评估 LLM 则需要应对开放式生成、主观质量判断、多步推理链,以及无处不在的基准污染(benchmark contamination)风险。本章对评估的图景进行系统性梳理:从评估类型的分类法与人工标注的机制,到排序度量的数学原理与 LLM 作为评判者(LLM-as-judge)的实践,再到那些会悄悄侵蚀评估流水线的陷阱。

## 为什么 LLM 评估很难

三个根本性挑战使 LLM 评估有别于经典机器学习评估:

1. **输出空间无界**。语言模型可以产生任意字符串,很少存在唯一的正确答案。
2. **质量是多维的**。有用性(helpfulness)、事实性(factuality)、安全性(safety)、连贯性(coherence)与风格(style)是相互独立的轴线,彼此之间可能存在权衡。
3. **评估本身就是一项语言任务**。判断一个回答是否优秀需要理解能力,这意味着评估同样容易陷入与生成相同的失败模式。

## 14.1 评估方案设计

在采集哪怕一个数据点之前,从业者都必须决定「测量什么」以及「如何测量」。一个有原则的分类法,可以避免「图方便选度量」而非「对齐部署目标选度量」这一常见错误。

### 14.1.1 评估类型分类法

**内在评估与外在评估(Intrinsic vs. Extrinsic Evaluation)**。内在评估(intrinsic evaluation)孤立地度量模型输出的属性,不涉及任何下游应用。留出语料上的困惑度(perplexity)、对照参考译文的 BLEU 分数、编码基准上的 pass@k 都属于内在评估。外在评估(extrinsic evaluation)度量模型对真实世界任务或系统的影响:将 LLM 接入客服流水线是否降低了工单升级率?编码助手是否提升了开发者的开发速度?

> **内在—外在鸿沟**
>
> 内在度量廉价且可复现,但往往与真实世界效用相关性很差。困惑度更低的模型未必更有用。外在度量昂贵且缓慢,但直接度量我们真正关心的事物。成熟的评估策略会以内在度量驱动快速迭代,以外在度量进行最终验证。

**自动评估与人工评估(Automatic vs. Human Evaluation)**。自动评估(automatic evaluation)使用确定性函数(BLEU、精确匹配)或学习到的模型(BERTScore、LLM-as-judge)在无需人工参与的情况下为输出打分。人工评估(human evaluation)则由标注员对模型输出评级或排序。表 14.1 总结了其中的权衡。

**表 14.1:评估方法的分类法及关键权衡。**

| 类型 | 成本 | 速度 | 可复现性 | 有效性 |
|---|---|---|---|---|
| 自动(基于规则) | 极低 | 极快 | 完美 | 低—中 |
| 自动(基于模型) | 低 | 快 | 高 | 中—高 |
| 众包人工 | 中 | 数天 | 中 | 中 |
| 专家人工 | 高 | 数周 | 低—中 | 高 |
| 外在 / A/B 测试 | 极高 | 数月 | 低 | 极高 |

**基于参考与无参考评估(Reference-Based vs. Reference-Free Evaluation)**。基于参考的度量(BLEU、ROUGE、BERTScore)将模型输出与一个或多个黄金标准参考(reference)进行对比。无参考的度量(困惑度、LLM-as-judge、人类偏好)在没有参考的情况下评估质量。当输出空间过大、无法穷尽地收集参考时(如开放式对话),无参考方法不可或缺。

### 14.1.2 何时用何法

> **对话助手的评估策略**
>
> - **开发阶段**:使用自动度量(困惑度、摘要子任务上的 ROUGE、工具使用上的 pass@k)进行快速迭代。在标准套件(MMLU、HellaSwag、HumanEval)上运行每夜基准测试。
> - **发布前阶段**:开展一项人类偏好研究,将新模型与上一个检查点(checkpoint)进行比较。使用 LLM-as-judge 在多样化的提示集上进行可扩展的两两比较。
> - **发布后阶段**:监控外在度量(用户满意度评分、任务完成率),并留意生产环境提示中的分布漂移(distribution shift)。

一个有用的决策框架:

- 若任务有明确正确答案(数学、代码、事实问答):使用精确匹配或基于执行的度量。
- 若任务是开放式的但有参考输出:将基于参考的度量作为下界,并以 LLM-as-judge 补充。
- 若任务是主观的(有用性、语气、创意):使用人工评估或一个校准良好的 LLM 评判者。
- 若任务涉及多步智能体行为:使用任务成功率与轨迹效率(见 14.6 节)。

## 14.2 评估的数据采集

高质量的评估数据是可信基准的基础。本节涵盖人工标注流水线的设计、标注质量的统计度量,以及众包与专家标注之间的取舍。

### 14.2.1 人工标注流水线

一个稳健的标注流水线由五个阶段组成:

1. **任务定义**。精确地规定标注任务:对什么打分、采用何种尺度、依据何种标准。此阶段的模糊性会蔓延成有噪声的标签。
2. **指南编写**。编写标注指南,提供覆盖边界情况的工作示例。在全面铺开前先与一小批试点人员进行迭代。
3. **标注员招募与培训**。挑选具有适当背景知识的标注员。举行一次校准会议,让标注员标注相同的示例并讨论分歧。
4. **质量控制**。将带有已知标签的黄金标准示例嵌入标注队列。对在黄金示例上准确率低于阈值的标注员进行标记。
5. **聚合**。对每个条目的多条标注,使用多数投票、平均或概率模型(例如 Dawid–Skene)进行合并。

### 14.2.2 标注者间一致性

原始一致率(所有标注者意见一致的条目所占比例)是一个不够格的度量,因为它没有考虑偶然一致。两种标准的「经偶然校正」的度量是 Cohen's $\kappa$ [252](两名标注者)和 Fleiss' $\kappa$ [253](多名标注者)。

**Cohen's Kappa**。给定两名标注者将 $N$ 个条目标注到 $k$ 个类别中,令 $p_o$ 为观测到的一致率,$p_e$ 为独立假设下的期望一致率:

$$
\kappa = \frac{p_o - p_e}{1 - p_e} \tag{14.1}
$$

其中

$$
p_o = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\text{标注者 1 与标注者 2 在条目 } i \text{ 上意见一致}] \tag{14.2}
$$

且

$$
p_e = \sum_{c=1}^{k} p_{1c} \cdot p_{2c} \tag{14.3}
$$

其中 $p_{jc}$ 是标注者 $j$ 将条目归入类别 $c$ 的比例。Cohen's $\kappa$ 取值范围从 $-1$(完全不一致)经 $0$(偶然一致)到 $1$(完全一致)。高于 0.6 的值通常被认为可接受;高于 0.8 为强一致。

**Fleiss' Kappa**。对 $n$ 名标注者将 $N$ 个条目标注到 $k$ 个类别中,令 $n_{ij}$ 为将条目 $i$ 归入类别 $j$ 的标注者人数。定义:

$$
\bar{P}_i = \frac{1}{n(n-1)} \sum_{j=1}^{k} n_{ij}(n_{ij} - 1), \qquad \bar{P} = \frac{1}{N} \sum_{i=1}^{N} \bar{P}_i \tag{14.4}
$$

$$
\bar{P}_e^{j} = \frac{1}{Nn} \sum_{i=1}^{N} n_{ij}, \qquad P_e = \sum_{j=1}^{k} \left(\bar{P}_e^{j}\right)^{2} \tag{14.5}
$$

$$
\kappa_F = \frac{\bar{P} - P_e}{1 - P_e} \tag{14.6}
$$

> **Kappa 的局限**
>
> Kappa 对类别的普遍程度(prevalence)敏感:当某一类别占主导时,即便原始一致率很高,kappa 也可能很低(即「kappa 悖论」)。对于序数尺度,加权 kappa(按分歧的距离成比例地施加惩罚)更为合适。在 LLM 评估中,评分常常采用 1–5 的李克特(Likert)量表,务必报告加权 kappa。

### 14.2.3 标注指南设计

有效的标注指南具备若干共性:

- **可操作的(operationalised)标准**。把「有帮助」这类模糊措辞替换为具体、可观察的行为:「该回答直接切中用户的问题,并提供了完成所述任务所需的全部信息。」
- **工作示例**。每个评分等级至少提供两个示例,包括边界情况。
- **决策树**。对于复杂任务,一张引导标注者通过一连串二元决策的流程图可降低认知负荷并提升一致性。
- **明确范围**。声明标注者不应考虑什么(例如:「不要因风格偏好而扣分;仅关注事实准确性」)。

### 14.2.4 众包与专家标注

**表 14.2:LLM 评估中众包标注与专家标注的对比。**

| 维度 | 众包(Crowdsourcing) | 专家标注(Expert Annotation) |
|---|---|---|
| 单条成本 | 低($0.01$–$0.10$) | 高($1$–$50$) |
| 吞吐量 | 极高 | 低 |
| 领域知识 | 低 | 高 |
| 一致性 | 不一 | 高 |
| 适用任务 | 简单偏好、流畅度 | 技术准确性、安全性 |
| 平台 | MTurk、Prolific、Scale AI | 领域专家、内部团队 |
| 质量控制 | 黄金示例、注意力检查 | 校准会议、同行评审 |

对于安全攸关的评估(例如检测有害输出、评估医疗建议),专家标注是不可妥协的。对于大规模偏好采集(例如构建奖励模型训练集),配以严格质量控制的众包往往是唯一可行的选项。

## 14.3 用于评估的合成数据生成

人工标注既昂贵又缓慢。合成数据生成(synthetic data generation)利用 LLM 自身大规模生产评估数据。本节介绍主要的范式。

### 14.3.1 用于校准的 LLM-as-Judge

当用 LLM 生成评估标签时,校准(calibration)至关重要:评判者的分数必须与人类判断对齐。令 $h_i \in [0, 1]$ 为条目 $i$ 的人类偏好分,$\hat{h}_i$ 为评判者预测的分。校准误差用期望校准误差(Expected Calibration Error,ECE)[254] 度量:

$$
\mathrm{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left|\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\right| \tag{14.7}
$$

其中 $B_b$ 是第 $b$ 个置信度分箱,$\mathrm{acc}(B_b)$ 是该箱中评判者与人类一致的条目所占比例,$\mathrm{conf}(B_b)$ 是该箱中评判者的平均置信度。

一个校准良好的评判者满足 $\mathbb{E}[\hat{h}_i \mid \hat{h}_i = p] = p$ 对所有 $p \in [0, 1]$ 成立。校准可通过温度缩放(temperature scaling)改进:将评判者的原始 logit $z$ 替换为 $z/T$,其中 $T$ 在留出校准集上调优,以最小化负对数似然。

### 14.3.2 Self-Instruct

Self-Instruct [255] 从一组人工编写的种子任务自举(bootstrap)出指令跟随数据。其算法:

1. 维护一个任务池,初始化为 175 个种子任务。
2. 从池中采样 8 个任务;将它们作为少样本(few-shot)示例提示 LLM 生成新任务。
3. 过滤生成任务:移除近似重复项(与任何已有任务的 ROUGE-L 相似度 > 0.7),将其分类为「分类任务」或「非分类任务」,并生成输入—输出实例。
4. 将通过审核的任务加入池中。
5. 重复,直至达到目标池规模。

```text
Self-Instruct 提示模板

system_prompt = """
Come up with a series of tasks:
Task 1: { seed_task_1_instruction }
Task 2: { seed_task_2_instruction }
...
Task 8: { seed_task_8_instruction }
Task 9:"""
```

模型补全该提示,生成一条新的任务指令。随后用另一个单独的提示为新任务生成输入—输出对。

### 14.3.3 Evol-Instruct

Evol-Instruct [256] 通过迭代重写指令使其更复杂或更多样,来演化一个种子指令集。应用两种演化算子:

- **深度演化(in-depth evolution)**:增加约束、增加推理步骤、把抽象具体化、加深领域知识要求。
- **广度演化(in-breadth evolution)**:在一个相关但不同的主题上生成新指令,以提升主题多样性。

一条指令若能通过淘汰筛选(elimination filter)即被接受:演化后的指令不能是简单拷贝,不能包含「I'm sorry」或类似拒绝措辞,且不能比原指令更短。

### 14.3.4 Constitutional AI 数据生成

Constitutional AI(CAI)[129] 通过让模型依据一组原则(即「宪法」)批判并修订自身输出来生成偏好数据。其流水线:

1. **监督学习阶段**:采样一个有害提示,生成一个初始回答,然后提示模型依据某条宪法原则对回答进行批判并修订。将修订后的回答用作监督微调(Supervised Fine-Tuning, SFT)的目标。
2. **RL 阶段**:生成成对的回答(原始版 vs. 修订版),让模型标注哪个更符合宪法,并在这些标签上训练一个偏好模型(preference model)。将该偏好模型作为 RLHF 的奖励信号。

这种方法在无需对有害内容进行人工标注的情况下生成偏好数据,从而减少标注者暴露于令人痛苦的材料。

### 14.3.5 用于评估数据的蒸馏

一个强大的教师模型(teacher model,例如 GPT-4)可以为训练一个更小的评判者模型生成高质量的评估数据。蒸馏(distillation)流水线:

1. 采集一组多样化的提示与模型回答。
2. 用教师模型生成详细评判(分数 + 理由)。
3. 在(prompt,response,judgment)三元组上微调一个更小的模型。
4. 在留出的人工标注上验证学生评判者。

> **蒸馏偏差**
>
> 从单一教师蒸馏得到的学生评判者会继承教师的偏差,包括冗长偏差(偏好更长的回答)、自我增强偏差(若教师同时是被评估的模型)以及位置偏差。务必对照独立的人工标注来验证蒸馏得到的评判者。

### 14.3.6 竞技场式两两生成

Chatbot Arena [257] 通过一个众包对战平台生成评估数据:用户提交提示,并就两个匿名模型回答中哪个更优进行投票。由此产生一个大规模、天然多样、由两两偏好构成的数据集。其关键设计选择:

- **匿名化**:隐藏模型身份以防止品牌偏差。
- **用户提交的提示**:确保提示的多样性与真实世界相关性。
- **平局处理**:用户可以宣布平局,或指出两个回答都很差。
- **去重**:过滤近似重复的提示,以防常见查询被过度代表。

## 14.4 排序任务的度量

当目标是按质量对模型排序时,两两比较数据比绝对分数更可靠。本节推导 LLM 评估中使用的主要排序系统。

### 14.4.1 ELO 评分系统

ELO 系统 [258] 最初为国际象棋开发,它给每个选手(模型)赋予一个标量评分 $R$,使得选手 A 对选手 B 的期望得分为:

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}} \tag{14.8}
$$

**推导**。ELO 模型假设每位选手在某局对局中的表现是一个服从逻辑斯蒂分布(logistic distribution)的随机变量,以其评分为中心。A 击败 B 的概率为:

$$
P(A \succ B) = \sigma\!\left(\frac{R_A - R_B}{s}\right) = \frac{1}{1 + e^{-(R_A - R_B)/s}} \tag{14.9}
$$

其中 $s = 400/\ln(10) \approx 173.7$ 是一个尺度参数,其取值使得 400 分的差距对应 $10 : 1$ 的赔率比。在每局以结果 $S_A \in \{0, 0.5, 1\}$(负、平、胜)结束后,评分更新为:

$$
R_A \leftarrow R_A + K(S_A - E_A), \qquad R_B \leftarrow R_B + K(S_B - E_B) \tag{14.10}
$$

其中 $K$ 是控制学习率的 K 因子(K-factor)。在 Chatbot Arena 中使用 $K = 4$。

> **ELO 直觉**
>
> ELO 是对逻辑斯蒂模型下观测结果对数似然的一次随机梯度下降更新。每局对局提供一个有噪声的梯度信号;K 因子控制步长。大的 $K$ 适应快但有噪声;小的 $K$ 稳定,但反映真实实力变化较慢。

**ELO 的自助法置信区间**。由于 ELO 评分取决于对局被处理的顺序,置信区间通过自助法(bootstrap)重采样计算:对对战日志有放回地重采样 $B = 1000$ 次,每次从头重新计算 ELO 评分,并将第 2.5 与第 97.5 百分位数报告为 95% 置信区间。

### 14.4.2 Bradley–Terry 模型

Bradley–Terry(BT)模型 [198] 是 ELO 的一种最大似然替代方案。给定 $n$ 个强度参数为 $\beta_1, \dots, \beta_n > 0$ 的模型,模型 $i$ 击败模型 $j$ 的概率为:

$$
P(i \succ j) = \frac{\beta_i}{\beta_i + \beta_j} \tag{14.11}
$$

给定一组两两结果 $\{(i_k, j_k, y_k)\}_{k=1}^{M}$,其中 $y_k = 1$ 若 $i_k$ 击败 $j_k$,否则 $y_k = 0$,其对数似然为:

$$
\ell(\beta) = \sum_{k=1}^{M} \left[ y_k \log\frac{\beta_{i_k}}{\beta_{i_k} + \beta_{j_k}} + (1 - y_k) \log\frac{\beta_{j_k}}{\beta_{i_k} + \beta_{j_k}} \right] \tag{14.12}
$$

MLE $\hat{\beta}$ 通过迭代缩放(iterative scaling)或梯度上升求得。BT 模型仅在一个乘性常数范围内可辨识;一种常见的归一化是 $\sum_i \log \beta_i = 0$。在对数空间中令 $\theta_i = \log \beta_i$ 则得:

$$
P(i \succ j) = \sigma(\theta_i - \theta_j) \tag{14.13}
$$

这等价于一个带有「条目专属截距」的逻辑回归。当完整的对战历史可得时,BT 模型优于 ELO,因为它同时使用所有数据,而非按顺序逐局处理。

### 14.4.3 TrueSkill

TrueSkill [259] 是一种贝叶斯技能评分系统,它将每位选手的技能建模为高斯随机变量 $s_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$。选手 $i$ 在一局对局中的表现为 $p_i = s_i + \epsilon_i$,其中 $\epsilon_i \sim \mathcal{N}(0, \beta^2)$ 是对局专属噪声。当 $p_i > p_j$ 时,选手 $i$ 击败选手 $j$。

在观测到 $i \succ j$ 之后,后验更新通过期望传播(expectation propagation, EP)计算。胜者的关键更新方程为:

$$
\mu_i \leftarrow \mu_i + \frac{\sigma_i^2}{c} \cdot v\!\left(\frac{\mu_i - \mu_j}{c}\right) \tag{14.14}
$$

$$
\sigma_i^2 \leftarrow \sigma_i^2 \left[ 1 - \frac{\sigma_i^2}{c^2} \cdot w\!\left(\frac{\mu_i - \mu_j}{c}\right) \right] \tag{14.15}
$$

其中 $c = \sqrt{2\beta^2 + \sigma_i^2 + \sigma_j^2}$,而 $v(t) = \phi(t)/\Phi(t)$、$w(t) = v(t)(v(t) + t)$ 是截断高斯修正因子($\phi$ 与 $\Phi$ 分别为标准正态的 PDF 与 CDF)。TrueSkill 的不确定性估计 $\sigma_i$ 对于识别那些需要更多评估数据的模型尤为有用。

### 14.4.4 带置信区间的胜率

最简单的排序度量是胜率(win rate):在两两比较中偏好模型 A 的比例。给定 $n$ 次比较中有 $w$ 次获胜,胜率为 $\hat{p} = w/n$。Wilson 分数置信区间 [260] 优于朴素的 Wald 区间,因为它在 $p = 0$ 与 $p = 1$ 附近覆盖更好:

$$
\mathrm{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \tag{14.16}
$$

其中 95% 区间取 $z = 1.96$。对于多方比较,胜率应针对一个固定的基线模型计算,以确保可比性。

### 14.4.5 Chatbot Arena 方法论

Chatbot Arena [257] 将上述要素组合为一个生产级评估系统:

1. 用户提交提示,并收到来自两个匿名模型的回答。
2. 用户为偏好的回答投票(或宣布平局)。
3. 使用 BT 模型聚合投票,生成排行榜。
4. 为每个模型的分数报告自助法置信区间。
5. 置信区间相互重叠的模型被视为在统计上不可区分。

截至 2024 年,Chatbot Arena 已收集超过一百万张人类偏好选票,是公开可得的最大 LLM 偏好数据集。

## 14.5 生成任务的度量

生成度量用于在任务带有参考答案或定义明确的正确性标准时,量化模型输出的质量。

### 14.5.1 BLEU

BLEU(Bilingual Evaluation Understudy)[261] 度量假设 $h$ 与一个或多个参考 $R$ 之间的 n-gram 精确率(precision):

$$
\mathrm{BLEU} = \mathrm{BP} \cdot \exp\!\left( \sum_{n=1}^{N} w_n \log p_n \right) \tag{14.17}
$$

其中 $p_n$ 是修正后的 n-gram 精确率,$w_n = 1/N$ 是均匀权重,$\mathrm{BP}$ 是简短惩罚(brevity penalty):

$$
\mathrm{BP} = \begin{cases} 1 & \text{若 } |h| > |r| \\ e^{1 - |r|/|h|} & \text{若 } |h| \le |r| \end{cases} \tag{14.18}
$$

其中 $|r|$ 为最近参考的长度。修正后的 n-gram 精确率将每个 n-gram 计数截断为它在任何参考中的最大计数:

$$
p_n = \frac{\sum_{\text{ngram} \in h} \min(\mathrm{count}(\text{ngram}, h),\, \max_{r \in R} \mathrm{count}(\text{ngram}, r))}{\sum_{\text{ngram} \in h} \mathrm{count}(\text{ngram}, h)} \tag{14.19}
$$

> **BLEU 的局限**
>
> BLEU 是为带有多条参考的机器翻译设计的。对于只有单一参考的开放式生成,即便输出质量很高,BLEU 分数也常常接近零。BLEU 无法捕捉语义相似度,会对合法的改写加以惩罚,并且对分词(tokenisation)敏感。仅当存在多条多样化参考、且任务输出多样性较低时,才使用 BLEU。

### 14.5.2 ROUGE

ROUGE(Recall-Oriented Understudy for Gisting Evaluation)[262] 是一族面向召回(recall)的度量,专为摘要设计:

$$
\mathrm{ROUGE\text{-}N} = \frac{\sum_{r \in R} \sum_{\text{ngram} \in r} \min(\mathrm{count}(\text{ngram}, h),\, \mathrm{count}(\text{ngram}, r))}{\sum_{r \in R} \sum_{\text{ngram} \in r} \mathrm{count}(\text{ngram}, r)} \tag{14.20}
$$

$$
\mathrm{ROUGE\text{-}L} = \frac{\mathrm{LCS}(h, r)}{|r|} \tag{14.21}
$$

其中 $\mathrm{LCS}$ 表示最长公共子序列(longest common subsequence)。ROUGE-1 与 ROUGE-2 度量一元(unigram)与二元(bigram)召回;ROUGE-L 捕捉句子级结构。F-measure 变体平衡精确率与召回:

$$
\mathrm{ROUGE\text{-}N_F} = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 P + R} \tag{14.22}
$$

其中取 $\beta = 1$ 以同等加权。

### 14.5.3 BERTScore

BERTScore [263] 利用来自预训练 BERT 模型的上下文嵌入(contextual embedding)计算词元级相似度。给定假设词元 $\hat{x} = \langle \hat{x}_1, \dots, \hat{x}_m \rangle$ 与参考词元 $x = \langle x_1, \dots, x_n \rangle$,其嵌入分别为 $\hat{e}_i$ 与 $e_j$:

$$
R_{\mathrm{BERT}} = \frac{1}{|x|} \sum_{x_j \in x} \max_{\hat{x}_i \in \hat{x}} \frac{\hat{e}_i^{\top} e_j}{\lVert \hat{e}_i \rVert \lVert e_j \rVert} \tag{14.23}
$$

$$
P_{\mathrm{BERT}} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_i \in \hat{x}} \max_{x_j \in x} \frac{\hat{e}_i^{\top} e_j}{\lVert \hat{e}_i \rVert \lVert e_j \rVert} \tag{14.24}
$$

$$
F_{\mathrm{BERT}} = \frac{2 \cdot P_{\mathrm{BERT}} \cdot R_{\mathrm{BERT}}}{P_{\mathrm{BERT}} + R_{\mathrm{BERT}}} \tag{14.25}
$$

BERTScore 与人类判断的相关性优于 BLEU 与 ROUGE,尤其对于改写以及语义等价但词表不同的输出。使用逆文档频率(inverse document frequency, IDF)的重要性加权可进一步提升相关性:

$$
R^{\mathrm{idf}}_{\mathrm{BERT}} = \frac{\sum_{x_j \in x} \mathrm{idf}(x_j) \max_{\hat{x}_i} \cos(\hat{e}_i, e_j)}{\sum_{x_j \in x} \mathrm{idf}(x_j)} \tag{14.26}
$$

### 14.5.4 METEOR

METEOR [264] 针对 BLEU 的召回盲点,在一元匹配上计算一个 F 分数,并附加用于词干化(stemming)与同义词匹配的模块:

$$
\mathrm{METEOR} = F_{\mathrm{mean}} \cdot (1 - \mathrm{Pen}) \tag{14.27}
$$

其中 $F_{\mathrm{mean}} = \frac{10 P R}{R + 9 P}$(召回加权的调和均值),碎片化惩罚(fragmentation penalty)$\mathrm{Pen} = 0.5 \cdot (c/u_m)^3$ 对非连续匹配加以惩罚($c$ = 块数,$u_m$ = 匹配的一元数)。

### 14.5.5 困惑度

困惑度(perplexity)度量语言模型对留出文本序列 $w_1, w_2, \dots, w_T$ 的预测好坏:

$$
\mathrm{PPL}(w_{1:T}) = \exp\!\left( -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(w_t \mid w_{1:t-1}) \right) \tag{14.28}
$$

困惑度越低,预测性能越好。困惑度适用于在同一分词与测试集上比较模型,但对于词表或分词器不同的模型则无法直接比较。就评估而言,困惑度最适合用作健全性检查(sanity check)以及检测分布漂移。

### 14.5.6 用于代码的 Pass@k

对于代码生成,功能正确性通过针对测试用例执行生成代码来度量。pass@k 度量 [265] 估计 $k$ 个生成样本中至少有一个通过全部测试的概率:

$$
\mathrm{pass@k} = \mathbb{E}_{\mathrm{problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right] \tag{14.29}
$$

其中 $n$ 是每个问题生成的样本总数,$c$ 是通过测试的样本数。该无偏估计量避免了朴素估计量(恰好采样 $k$ 个解并检查是否有通过的)的高方差。实践中生成 $n = 200$ 个样本,并报告 pass@1、pass@10、pass@100。

```python
# Pass@k 计算
import numpy as np
from scipy.special import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k.
    Args:
        n: total samples generated per problem
        c: number of samples that pass all tests
        k: number of samples to consider
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True)

# 示例:200 个样本,15 个通过,计算 pass@1、pass@10、pass@100
for k in [1, 10, 100]:
    score = pass_at_k(n=200, c=15, k=k)
    print(f"pass@{k}: {score :.4f}")
# pass@1:   0.0750
# pass@10:  0.5391
# pass@100: 0.9999
```

### 14.5.7 精确匹配与 F1

对于抽取式问答(如 SQuAD),有两种标准度量:

- **精确匹配(Exact Match, EM)**:在归一化(小写化、移除冠词与标点)后,预测答案字符串是否与任一黄金答案完全一致的二元指示。
- **词元级 F1**:将预测与黄金答案视作词袋(bags of tokens),计算 F1 分数:

$$
F_1 = \frac{2 \cdot |\mathrm{pred} \cap \mathrm{gold}|}{|\mathrm{pred}| + |\mathrm{gold}|} \tag{14.30}
$$

对于多答案情形,报告所有黄金答案中的最大 F1。

**表 14.3:生成度量汇总:适用性与关键属性。**

| 度量 | 任务 | 无参考? | 与人类相关性 |
|---|---|---|---|
| BLEU | 翻译 | 否 | 低—中 |
| ROUGE | 摘要 | 否 | 中 |
| BERTScore | 通用 NLG | 否 | 高 |
| METEOR | 翻译 | 否 | 中—高 |
| Perplexity | 语言模型质量 | 是 | 低 |
| Pass@k | 代码生成 | 否(测试) | 极高 |
| Exact Match | 抽取式 QA | 否 | 极高 |
| Token F1 | 抽取式 QA | 否 | 高 |

## 14.6 智能体任务的度量

智能体式 LLM 在环境中运行,采取一系列动作,并必须完成多步任务。标准的生成度量不足以胜任;智能体评估需要能够捕捉任务完成度、效率以及中间步骤质量的度量。

### 14.6.1 任务成功率

智能体任务的首要度量是任务成功率(task success rate, TSR):智能体达成指定目标状态的条目所占比例:

$$
\mathrm{TSR} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \mathbf{1}[\text{goal}(\tau)\text{ 已达成}] \tag{14.31}
$$

目标达成通常由一个确定性预言机(oracle)验证(例如检查数据库状态、文件系统状态或测试用例执行)。对于可部分给分的任务,可定义一种分级成功度量:

$$
\mathrm{TSR}_{\mathrm{graded}} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \mathrm{score}(\tau) \in [0, 1] \tag{14.32}
$$

### 14.6.2 轨迹效率

一个成功的智能体应当以最少的冗余动作完成任务。轨迹效率(trajectory efficiency)度量最优轨迹长度与智能体实际轨迹长度之比:

$$
\eta = \frac{L^*}{L_{\mathrm{agent}}} \tag{14.33}
$$

其中 $L^*$ 是最短成功轨迹的长度(由预言机或人类专家计算),$L_{\mathrm{agent}}$ 是智能体所采取的动作数。$\eta \in (0, 1]$,$\eta = 1$ 表示最优效率。对于失败轨迹,$\eta = 0$。

一个互补的度量是冗余率(redundancy rate):不出现在任何最优轨迹中的智能体动作所占比例。

### 14.6.3 工具使用准确率

对于调用外部工具(API、代码解释器、搜索引擎)的智能体,工具使用准确率(tool-use accuracy)度量工具调用的正确性:

$$
\mathrm{TUA} = \frac{\#\ \text{正确的工具调用}}{\#\ \text{工具调用总数}} \tag{14.34}
$$

一次工具调用是正确的,当且仅当(a)选对了工具、(b)参数有效、(c)在轨迹中恰当的时机发起调用。对于「工具选对但参数错误」的情形,可给予部分分。

### 14.6.4 多步推理准确率

对于需要推理链的任务(例如多跳问答(multi-hop QA)、数学问题求解),步骤级准确率(step-level accuracy)度量正确推理步骤所占比例:

$$
\mathrm{SRA} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \frac{1}{|S_\tau|} \sum_{s \in S_\tau} \mathbf{1}[s\ \text{是正确的}] \tag{14.35}
$$

其中 $S_\tau$ 是轨迹 $\tau$ 中的推理步骤集合。步骤正确性可由过程奖励模型(process reward model, PRM)或人工标注验证。

### 14.6.5 SWE-bench 方法论

SWE-bench [266] 在真实世界软件工程任务上评估 LLM:给定一个 GitHub issue 描述与仓库代码库,模型必须生成一个解决该 issue 的补丁。评估流程如下:

1. 向模型给出 issue 描述与相关代码上下文。
2. 模型生成一个补丁(unified diff 格式)。
3. 将补丁应用到仓库。
4. 执行该仓库的测试套件;若全部测试通过,则任务成功。

首要度量是 % Resolved:生成的补丁通过全部测试的 issue 所占比例。SWE-bench Verified 是一个由人工标注员验证过可解且无歧义的 500 题精选子集。SWE-bench Lite 是一个用于更快评估的 300 题子集。

> **SWE-bench 关键统计(截至 2024 年)**
>
> - 完整基准:来自 12 个流行 Python 仓库的 2,294 个任务。
> - 最佳开源智能体:SWE-bench Verified 上约 43% 解决率。
> - 人类表现:约 87% 解决率(每题 15 分钟)。
> - 评估成本:基于 API 的模型约 $0.25/题。

### 14.6.6 WebArena 方法论

WebArena [267] 在一个沙箱化浏览器环境中,针对真实的网页导航任务评估智能体。该基准包含跨五个网页应用(电子商务、社交论坛、协作开发、内容管理、地图)的 812 个任务。评估:

- **功能性评估**:通过检查应用状态来验证任务结果(例如「该商品是否已加入购物车?」「该帖子是否已创建?」)。
- **基于 URL 的评估**:对于导航任务,将最终 URL 与期望 URL 进行对比。
- **基于程序的评估**:由自定义评估脚本检查复杂条件(例如「价格是否低于 $50?」)。

首要度量为任务成功率。人类表现约为 78%;最先进智能体约达 35–45%。

## 14.7 LLM 作为评判者

LLM-as-judge [257] 使用一个有能力的 LLM 来评估其他(或同一个)LLM 的输出。该方法可在无需人工标注的情况下扩展到大规模评估集,并能为其判断提供详细理由。

**表 14.4:智能体评估基准对比。**

| 基准 | 领域 | 任务数 | 评估方法 | SOTA(%) |
|---|---|---|---|---|
| SWE-bench | 软件工程 | 2,294 | 测试执行 | ~43 |
| SWE-bench Lite | 软件工程 | 300 | 测试执行 | ~50 |
| WebArena | 网页导航 | 812 | 状态/URL/程序 | ~40 |
| ALFWorld [268] | 家务任务 | 3,553 | 模拟器状态 | ~90 |
| AgentBench [269] | 多领域 | 1,091 | 任务专属 | ~45 |

### 14.7.1 设置与提示模板

评判者被给到一个提示、一个或多个模型回答,以及一份评估量规(rubric)。三种常见格式:

**逐点评分(pointwise scoring)**。评判者对单个回答给出绝对分数:

```text
逐点评判提示

POINTWISE_PROMPT = """
You are an expert evaluator. Rate the following response on a scale
of 1-10 for helpfulness, accuracy, and clarity.
[Question]
{question}
[Response]
{response}
Provide your evaluation in the following format:
Reasoning: <step-by-step analysis>
Score: <integer from 1 to 10>
"""
```

**两两比较(pairwise comparison)**。评判者比较两个回答并选出更优者:

```text
两两评判提示

PAIRWISE_PROMPT = """
You are an expert evaluator. Compare the two responses below and
determine which is better. Consider helpfulness, accuracy, and depth
of explanation.
[Question]
{question}
[Response A]
{response_a}
[Response B]
{response_b}
Output exactly one of: [[A]], [[B]], or [[C]] (tie).
Reasoning: <your analysis>
Verdict: <[[A]], [[B]], or [[C]]>
"""
```

**参考引导评分(reference-guided scoring)**。向评判者提供一份参考答案,并据此对回答评分。这对于评判者本身可能不具备可靠知识的事实性任务尤为有用。

### 14.7.2 位置偏差的缓解

LLM 评判者表现出位置偏差(position bias):对出现在某一特定位置(首位或末位)的回答存在系统性偏好。这种偏差可达 10–15 个百分点。缓解策略:

1. **交换增广(swap augmentation)**:以两种顺序(A vs. B 与 B vs. A)评估每一对。一致判断予以接受;不一致判断记为平局。
2. **校准提示(calibration prompting)**:明确指示评判者:「你的评估不应受到回答呈现顺序的影响。」
3. **集成评判(ensemble judging)**:使用多个具有不同位置排列的评判者,并聚合其裁决。
4. **强制思维链(chain-of-thought forcing)**:要求评判者在给出裁决前先产出详细理由,从而降低对表面位置线索的依赖。

> **冗长偏差(Verbosity Bias)**
>
> LLM 评判者也会表现出冗长偏差:更长的回答被系统性偏好,即便额外内容无关或重复。为缓解这一点,可指示评判者惩罚不必要的冗长,并聚焦于信息的质量而非数量。或者,在评判前将回答截断到固定长度。

### 14.7.3 多评判者小组

单一评判者可能带有系统性偏差。一个由来自不同模型家族的评判者构成的小组可提供更稳健的评估。给定 $J$ 名评判者的裁决 $v_1, \dots, v_J \in \{A, B, \text{tie}\}$,小组裁决由多数投票决定。小组一致率为:

$$
\mathrm{Agreement} = \frac{1}{\binom{J}{2}} \sum_{i < j} \mathbf{1}[v_i = v_j] \tag{14.36}
$$

对于一个三人评判小组:一致裁决(三人全同意)视为高置信;2–1 分裂视为中置信;三方平局视为低置信。

### 14.7.4 LLM 评判者的一致性度量

为验证一个 LLM 评判者,会将其裁决与留出集上的人工标注对比。关键度量:

- **一致率(agreement rate)**:评判者与人类意见一致的条目所占比例。
- **Cohen's $\kappa$**:经偶然校正的一致率(式 14.1)。
- **Spearman 的 $\rho$**:评判者分数与人类分数之间的秩相关,适用于序数评分。
- **Kendall 的 $\tau$**:另一种秩相关,对平局更稳健。

若一个评判者在代表性样本上对人工标注实现 $\kappa > 0.6$ 且一致率 $> 80\%$,则被视为可靠。

### 14.7.5 G-Eval 框架

G-Eval [270] 是一个面向基于 LLM 评估的结构化框架,它使用思维链(chain-of-thought, CoT)提示与词元概率加权来产生更可靠的分数。该框架:

1. **生成评估步骤**:提示 LLM 为评估任务生成一份详细的量规(例如:「列出你会采取哪些步骤来评估一篇摘要的连贯性」)。
2. **带概率加权打分**:对每个分值 $s \in \{1, 2, 3, 4, 5\}$,从评判者模型获得对数概率 $\log P_\theta(s \mid \text{prompt, steps, response})$。最终分数为概率加权平均:

$$
\text{G-Eval score} = \sum_{s=1}^{5} s \cdot \frac{e^{\log P_\theta(s)}}{\sum_{s'=1}^{5} e^{\log P_\theta(s')}} \tag{14.37}
$$

3. **归一化**:除以最大分数,将分数映射到 $[0, 1]$。

G-Eval 与人类判断的相关性高于直接提示,尤其对于连贯性、一致性等微妙维度,因为概率加权捕捉了评判者的不确定性,而非强制做出离散选择。

> **为什么 G-Eval 有效**
>
> 标准提示让评判者输出单个词元(例如「4」),这丢弃了模型的不确定性。G-Eval 读取所有评分词元上的概率分布,实际上计算了在评判者信念下的期望分数。这类似于使用后验分布的均值而非众数。

## 14.8 评估陷阱

即便精心设计的评估流水线也可能产生误导性结果。本节梳理最常见的失败模式。

### 14.8.1 基准污染

基准污染(benchmark contamination)在评估数据出现在模型训练集中时发生,既可能是直接(逐字收录),也可能是间接(改写或语义相近的内容)。被污染的模型获得虚高的分数,无法反映真实的泛化能力。

检测方法:

- **n-gram 重叠**:计算与训练语料有高 n-gram 重叠(例如 ROUGE-L > 0.8)的评估样本所占比例。
- **成员推断(membership inference)**:使用成员推断攻击估计每个评估样本位于训练集中的概率。
- **金丝雀串(canary strings)**:在评估样本中嵌入唯一的、随机生成的字符串,并检查模型是否能补全它们。
- **时间留出(temporal holdout)**:使用在模型训练截止日期之后创建的评估数据。

缓解:

- 维护一个永不公开发布的私有测试集。
- 定期用新示例刷新基准。
- 报告训练数据截止日期与去污(decontamination)流程。

### 14.8.2 对基准过拟合

即便没有直接污染,模型也可能通过反复评估与超参数调优而被隐式地针对特定基准优化。这是一种自适应过拟合(adaptive overfitting):基准向模型开发决策中泄漏了信息。

> **基准的生命周期**
>
> 随着研究界围绕某个基准做优化,其效用会随时间衰减。MMLU [271] 曾是对世界知识的严苛测试,如今已有模型取得接近人类的表现,然而这些模型在全新的知识任务上仍会失败。新的基准应被视作临时的信号源,而非永久的真值。

### 14.8.3 评估中的古德哈特定律

古德哈特定律(Goodhart's Law)指出:「当一个度量成为目标时,它就不再是一个好的度量。」[272] 在 LLM 评估中,这有几种表现形式:

- **奖励投机(reward hacking)**:用 RLHF 训练的模型学会钻奖励模型的空子,而非真正改进。一个模型可能学会产出冗长、听起来自信的回答,它们在奖励模型上得分很高,却在事实上不正确。
- **度量投机(metric gaming)**:为最大化 BLEU 或 ROUGE 而微调的模型可能产出在这些度量上得分很高、但对人类用处更小的输出。
- **评判者投机(judge gaming)**:以 LLM-as-judge 反馈训练的模型可能学会评判者的偏差(例如冗长偏差),而非真正提升质量。

**抵御古德哈特定律**

1. **度量多样化**:使用来自不同家族的多种度量;一个能投机于某度量的模型,通常无法同时投机于所有度量。
2. **留出评估**:维护一些未用于训练或模型选择的评估度量。
3. **人工抽查**:定期抽样模型输出供人工评审,独立于自动度量。
4. **对抗评估**:主动探测自动度量遗漏的失败模式。
5. **外在验证**:周期性地以外在结果验证内在度量。

### 14.8.4 其他陷阱

**提示敏感性(prompt sensitivity)**。LLM 的性能可能因评估提示的微小变化(例如添加「Think step by step」或改变答案格式)而剧烈波动。务必报告所用的确切提示,并考虑在多种提示变体上评估。

**聚合伪影(aggregation artefacts)**。对难度与分数分布不同的任务求平均,可能产生误导性的聚合度量。一个擅长简单任务却在困难任务上失败的模型,其平均分可能与一个表现均匀的模型相同。

**人工评估中的选择偏差(selection bias)**。人工评估者并非最终用户的随机样本。众包平台上的标注者在偏好、文化背景与领域知识上可能与目标用户群体不同。

**评估—部署错配(evaluation–deployment mismatch)**。评估提示往往比真实用户查询更短、更干净、更规整。在基准提示上表现良好的模型,在生产中那些嘈杂、含混、多轮的对话上可能显著退化。

> **评估设计的关键问题**
>
> 在部署评估流水线之前,请自问:
>
> 1. 评估度量是否与部署目标对齐?
> 2. 评估数据是否对目标分布具有代表性?
> 3. 是否已评估污染与过拟合风险?
> 4. 是否为所有度量报告了置信区间?
> 5. 评估是否可复现(固定随机种子、版本化的提示、公开测试集)?
> 6. 评估是否已对照人类判断或外在结果加以验证?
