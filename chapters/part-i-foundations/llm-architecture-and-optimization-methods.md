# 第 1 章 LLM 架构与优化方法

本节涵盖大语言模型(Large Language Model, LLM)的基础架构,以及让训练和推理变得高效的关键优化技术。各主题按课程顺序编排:我们先从 Transformer 本身讲起,然后依次介绍如何高效训练、如何低成本适配、如何压缩、如何扩展,以及如何加速其推理。

## 1.1 LLM 如何工作:直观概述

在深入架构细节之前,让我们先建立直觉,理解大语言模型如何把文本转换为文本。整个过程遵循一条简单的流水线:文本 → 词元(token)→ 表示 → 词元 → 文本。

![图 1.1:LLM 流水线——文本被分词为子词单元,转换为整数 ID,嵌入为稠密向量,经过 Transformer 层处理,投影到词表的 logits,最后解码回文本。虚线回路表示自回归生成——每个输出词元都被追加到输入中,用于下一次前向传播。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p36-01.png)

### 四个关键阶段

1. **分词(Tokenization)**:原始文本使用一个学习到的词表被切分为子词片段(既不是字符,也不是完整单词)。例如 "unhappiness" 可能被切成 `["un", "happiness"]` 或 `["unhapp", "iness"]`。
2. **嵌入(Embedding)**:每个词元 ID 在一张学习到的嵌入表中查找索引,生成一个位于 $\mathbb{R}^d$(通常 $d = 4096$)中的稠密向量。这些向量捕获语义含义——相似的词会得到相似的向量。
3. **上下文处理(Contextual Processing)**:Transformer 堆叠并行处理所有嵌入,使用自注意力(self-attention)让每个位置都能"读取"其他所有位置。经过 $L$ 层之后,每个位置的隐藏状态都编码了丰富的上下文信息。
4. **预测(Prediction)**:最终的隐藏状态被投影到整个词表上的概率分布,然后由一种解码(decoding)策略选出下一个词元。

## 1.2 分词

分词是关键的第一步,它把原始文本转换成语言模型所操作的离散符号。分词器(tokenizer)的选择会直接影响模型质量、多语言能力和计算效率。

### 为什么用子词?

字符级模型需要非常长的序列(注意力代价高昂)。词级模型无法处理罕见词或新词。子词分词达成了理想的折中:常见词成为单个词元(`"the"` → `[the]`),罕见词分解为已知片段(`"cryptocurrency"` → `["crypt", "ocur", "rency"]`),而且词表规模保持可控(32K–128K 个词元)。

### 1.2.1 为什么不用字符或单词?

表 1.1:不同分词粒度的权衡。

| 粒度 | 词表大小 | 序列长度 | 问题 |
|---|---|---|---|
| 字符 | ~256 | 非常长 | 注意力开销 $O(n^2)$;难以学习长程语义 |
| 词 | ~500K+ | 短 | 无法处理罕见/新词;嵌入表巨大 |
| 子词 | 32K–128K | 适中 | 最佳折中:序列短,词表开放 |

### 1.2.2 字节对编码(Byte-Pair Encoding, BPE)

BPE [24] 是 GPT、Llama、Mistral 以及大多数现代 LLM 所使用的主流分词算法。

**BPE 算法**

1. 从一个由单个字符(字节)组成的词表开始
2. 统计训练语料中所有相邻符号对的出现次数
3. 把出现频率最高的对合并为一个新符号
4. 重复步骤 2–3 共 $k$ 次迭代(直到达到目标词表大小)

图 1.2 描述了 BPE 的分词示例:从字符开始,算法迭代地合并出现频率最高的相邻对,直到该词变为单个词元,或词表预算耗尽。

### 1.2.3 其他分词方法

表 1.2:子词分词算法比较。

| 方法 | 使用者 | 核心思想 |
|---|---|---|
| BPE | GPT-4 [23]、Llama-3 [25]、Mistral [26] | 自底向上合并高频对;确定性 |
| WordPiece | BERT [27]、DistilBERT [28] | 类似 BPE,但最大化训练数据的似然 |
| Unigram LM | SentencePiece(T5 [29]、XLNet [30]) | 自顶向下:从大词表开始,按似然影响进行剪枝 |
| Byte-level BPE | GPT-2 [31]+ | 在原始字节上做 BPE(不会有未知词元);256 个基础词表 |

### 1.2.4 分词最佳实践

1. **词表大小很重要**:32K 是最低限度;128K 能带来更好的多语言覆盖和代码处理能力。Llama-3 使用 128K 个词元。
2. **特殊词元**:始终包含 `<bos>`、`<eos>`、`<pad>`、`<unk>`。对于指令微调(instruction-tuned)的模型,还要添加角色标记(`<|user|>`、`<|assistant|>`)。
3. **繁殖率(Fertility)**:衡量各语言中每个词对应的词元数。高繁殖率(一个词对应很多词元)表明该语言覆盖不佳。
4. **永远不要跨边界分词**:空格、标点和数字应当被一致地处理。大多数现代分词器会在词前加一个空格标记(`"the"`)以区分词首词元和接续词元。
5. **数字**:对算术任务,考虑按数字位分词。"2024" 切成 `["2", "0", "2", "4"]` 可以实现逐位推理。
6. **代码**:确保空白(缩进)被高效地分词。Llama-3 把连续的空格作为单个词元。

### 1.2.5 分词实践:HuggingFace 示例

`transformers` 库为所有分词器提供了统一接口。下面演示如何用现代 LLM 分词器进行编码和解码:

```python
from transformers import AutoTokenizer

# 加载 Llama-3 分词器(128K 词表,字节级 BPE)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
text = "Reinforcement learning optimizes long-term rewards."

# 编码:文本 -> 词元 ID
token_ids = tokenizer.encode(text)
print(token_ids)
# [128000, 29934, 262, 11008, 4815, 6900, 1317, 9860, 21845, 13]

# 解码单个词元,查看子词切分
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)
# ['<|begin_of_text|>', 'Re', 'inforce', 'ment', ' learning',
#  ' optimizes', ' long', '-term', ' rewards', '.']

# 解码回文本(往返还原)
reconstructed = tokenizer.decode(token_ids, skip_special_tokens=True)
assert reconstructed == text  # 完美重建

# 带注意力掩码的分词(用于带填充的批量输入)
batch = tokenizer(
    ["Short text.", "A much longer input sentence for comparison."],
    padding=True, return_tensors="pt"
)
print(batch.keys())
# dict_keys(['input_ids', 'attention_mask'])
```

代码清单 1.1:使用 HuggingFace Transformers 进行分词的编码/解码。

### 1.2.6 特殊词元与结构化提示

特殊词元(special tokens)是保留的词表条目,承载的是结构性含义而非语言内容。它们对于控制模型行为至关重要。

表 1.3:各 LLM 家族常见的特殊词元。

| 词元 | 别名 | 用途 |
|---|---|---|
| `<bos>` / `<\|begin_of_text\|>` | BOS | 标记序列起始 |
| `<eos>` / `<\|end_of_text\|>` | EOS | 标记序列结束;停止生成 |
| `<\|user\|>` | — | 标记对话中用户回合的开始 |
| `<\|assistant\|>` | — | 标记对话中助手回合的开始 |
| `<pad>` | PAD | 将批量填充到统一长度;在注意力中被掩蔽 |
| `<unk>` | UNK | 词表外占位符(使用 BPE 时很少见) |
| `[SEP]` | SEP | 分隔片段(BERT 风格) |
| `[CLS]` | CLS | 分类词元(BERT) |
| `[MASK]` | MASK | 用于 MLM 预训练的被掩蔽词元 |

**指令微调模型的角色标记。** 现代对话模型使用特殊词元来勾勒对话结构。这些词元并非被训练来承载语义含义——它们只是模型学习去解析的结构性分隔符:

```python
# Llama-3 对话模板
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain PPO in one sentence."},
]

# apply_chat_template 负责所有特殊词元的插入
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
#
# Explain PPO in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
#
```

代码清单 1.2:带特殊词元的对话模板(Llama-3 格式)。

**特殊词元最佳实践**

- **永远不要拆分特殊词元**:它们必须是原子的——确保你的分词器把它们当作单个单元,而非字符序列。
- **对特殊词元掩蔽损失**:在监督微调(Supervised Fine-Tuning, SFT)期间,不要在结构性词元(角色标记、分隔符)上计算损失。模型不应当"学习"去预测格式。

## 1.3 Transformer 架构

Transformer [6] 是所有现代 LLM 的基础。理解它的各个组件，对于掌握本指南后续的每一种优化与训练方法都至关重要。

### 1.3.1 整体结构

解码器架构（decoder-only transformer）按如下流程依次处理词元（token）：经过嵌入（embedding），反复堆叠的「注意力 + FFN」块，最后投影到词表的 logits。图 1.3 展示了完整的架构。

![图 1.3：解码器架构 Transformer 块（GPT 风格，Pre-Norm 变体）。每个子层（注意力、FFN）前接 LayerNorm，后接残差相加：x + SubLayer(LN(x))。这种 Pre-Norm 排序（被 Llama、GPT-3、Mistral 采用）无需预热即可稳定训练，而原始的 Post-Norm（在相加之后再做 LayerNorm）则不然。共堆叠 L 个相同块，最后接一个 LayerNorm 和到词表 logits 的线性投影。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p40-02.png)

### 1.3.2 原始的编码器-解码器 Transformer

Transformer 最初 [6] 是作为面向序列到序列任务（机器翻译、摘要）的编码器-解码器（encoder-decoder）架构被提出的。尽管现代 LLM 主要采用仅解码器（decoder-only）的变体（GPT 风格），但理解完整架构仍然必不可少，因为交叉注意力（cross-attention）和带掩码的自注意力（masked self-attention）——两者都源自这里——至今仍是基础构件。

**编码器（Encoder）。** 编码器双向地处理整个输入序列——每个词元都能注意到所有其他词元（没有因果掩码）。这会产生丰富的上下文表示 $H_\text{enc} \in \mathbb{R}^{n \times d}$，其中每个位置都编码了关于完整输入的信息：

- 输入：词元嵌入 + 正弦位置编码
- 每一层：多头自注意力 → 加 & 归一化（Add & Norm） → FFN → 加 & 归一化
- 无因果掩码：位置 $i$ 可以注意到所有位置 $1, \dots, n$
- 输出：完整输入序列的上下文表示

**解码器——带掩码的多头自注意力。** 解码器一次生成一个输出词元（自回归地）。为防止模型「看到未来」，解码器中的自注意力使用了因果掩码：

$$
\text{MaskedAttn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
\quad (1.1)
$$

其中掩码 $M$ 为：

$$
M_{ij} =
\begin{cases}
0 & \text{若 } i \geq j \text{（可以注意）} \\
-\infty & \text{若 } i < j \text{（未来词元——被屏蔽）}
\end{cases}
$$

**为何掩码很重要**

在训练时，解码器并行处理整个目标序列（teacher forcing，教师强制），但每个位置只能注意之前的位置，以保持自回归特性。掩码确保生成词元 $t$ 时只使用词元 $1, \dots, t-1$ 的信息。在推理时，词元逐个生成，因此掩码是隐式的——但在训练时它使得在保持因果性的同时进行并行计算成为可能。

**解码器——交叉注意力。** 在带掩码的自注意力之后，每个解码器层都会应用交叉注意力，此时解码器去注意编码器的输出表示。这就是解码器「读取」输入的机制：

$$
\text{CrossAttn}(Q_\text{dec}, K_\text{enc}, V_\text{enc}) = \text{softmax}\!\left(\frac{Q_\text{dec} K_\text{enc}^\top}{\sqrt{d_k}}\right) V_\text{enc}
\quad (1.2)
$$

- 查询（Queries）来自解码器的前一个子层（带掩码自注意力的输出）
- 键（Keys）和值（Values）来自编码器的最终输出 $H_\text{enc}$
- 不施加掩码——每个解码器位置都能注意到每个编码器位置
- 这使得解码器能够在每个生成步骤动态地聚焦于输入的不同部分（例如，在英译西时，生成「gato」时会注意到「cat」）

![图 1.4：原始 Transformer 架构（Vaswani 等人，2017）。编码器（左）用双向自注意力处理完整输入。解码器（右）使用带掩码的自注意力以及对编码器表示的交叉注意力，自回归地生成词元。虚线框表示重复的层块（×N）；灰线表示绕过各子层的残差连接。注意：原始工作采用 Post-Norm（残差相加之后再做 LayerNorm：LN(x + SubLayer(x))），这与使用 Pre-Norm 的现代 LLM 不同。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p41-03.png)

**完整的解码器层。** 每个解码器层包含三个子层（相比之下编码器只有两个）：

1. 带掩码的多头自注意力 + 残差 + LayerNorm
2. 多头交叉注意力（针对编码器输出） + 残差 + LayerNorm
3. 前馈网络 + 残差 + LayerNorm

**从编码器-解码器到仅解码器。** 现代 LLM（GPT、Llama、Qwen）只使用解码器，完全移除了编码器和交叉注意力层。关键洞见在于：对于生成式语言建模，单一的因果（带掩码）自注意力堆栈就足够了——模型学会在单次前向中既编码上下文又生成延续。这简化了架构、训练和推理，同时具有更好的可扩展性。编码器-解码器模型（T5、BART）在输入/输出结构截然不同的任务（翻译、摘要）中仍然有用武之地，而交叉注意力在多模态模型中重新出现——由视觉编码器向语言解码器提供键/值。

### 1.3.3 仅解码器 vs 编码器-解码器

现代 LLM 几乎全部使用仅解码器架构，但理解其与编码器-解码器设计的权衡能说明为什么会如此。

| 架构 | 示例 | 用途 |
|---|---|---|
| 仅解码器 | GPT-4 [23]、Llama [25]、Mistral [26]、Qwen [32] | 自回归生成；在对话/推理任务中占主导 |
| 编码器-解码器 | T5 [29]、BART [33]、Flan-T5 [34] | Seq2seq（翻译、摘要）；现已较少使用 |
| 仅编码器 | BERT [27]、RoBERTa [35] | 分类/嵌入；不用于生成 |

**为何仅解码器胜出**

仅解码器模型更简单（一个模型、一个损失），扩展性更好（所有参数都参与生成），并支持统一的训练（预训练 = 下一词元预测 = 微调目标）。编码器-解码器模型在纯生成任务上会把容量浪费在编码器上。

### 1.3.4 嵌入：从离散词元到连续空间

在任何注意力或计算发生之前，transformer 必须把离散的词元 ID 转换为神经网络能够处理的连续向量。这正是嵌入层的职责。

**什么是嵌入？** 嵌入（embedding）是离散符号的一种学得的稠密向量表示。与其把单词「king」表示为一个大小为 $|V| = 128{,}000$ 的 one-hot 向量（几乎全为 0），不如把它表示为一个位于 $\mathbb{R}^d$（例如 $d = 4096$）中的紧凑向量，用以捕捉其含义。关键洞见：相似的概念会得到相近的向量。在一个训练良好的嵌入空间中：

- 「king」和「queen」很接近（都与王室相关）
- 「king」和「bicycle」相距甚远（互不相关）
- 向量运算可以捕捉关系：

$$
\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}
$$

**嵌入表。** 在实践中，嵌入层只是一个矩阵 $E \in \mathbb{R}^{|V| \times d}$，其中第 $i$ 行存储词元 $i$ 的嵌入向量：

$$
\text{embed}(x_t) = E[x_t] \in \mathbb{R}^d
\quad (1.3)
$$

对于一个词元 ID 序列 $[x_1, x_2, \dots, x_n]$，嵌入就是一次简单的查表（索引操作）：

$$
H_0 = [E[x_1];\; E[x_2];\; \dots;\; E[x_n]] \in \mathbb{R}^{n \times d}
$$

**Transformer 中的嵌入表**

- 大小：$|V| \times d$。对于 Llama-3：$128{,}256 \times 4096 = 525\text{M}$ 参数（占 8B 模型的 6.5%）。
- 初始化：随机（Xavier/正态），然后通过反向传播学习。
- 权重绑定（Weight tying）：许多模型把嵌入矩阵与输出投影头共享：$W_\text{head} = E^\top$。这既节省参数，又构建了对称的编码-解码结构。
- 输入：词元 ID（整数） → 输出：$\mathbb{R}^d$ 中的稠密向量。
- 梯度流：训练时，只有与当前 batch 中词元对应的行才会接收梯度更新（稀疏更新）。

![图 1.5：嵌入空间可视化（2D 投影）：语义相似的词聚集在一起。嵌入表在预训练期间学会这些位置，纯粹从文本的共现模式中捕捉含义。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p43-04.png)

**嵌入为何有效**

嵌入表与模型其余部分一起端到端地学习。由于模型被训练来预测下一个词元，它必须学会这样的表示：出现在相似上下文中的词元会得到相似的向量。这就是分布假说（distributional hypothesis）：「你可通过一个词的同伴来认识它」[36]。嵌入层把这种统计结构压缩进了稠密的几何之中。

**各向异性问题（Anisotropy）。** 当使用预训练嵌入（例如来自 BERT 或 GPT-2）于检索（RAG）或冷启动推荐系统等下游任务时，会出现一个关键问题：所学得的表示是高度各向异性的——它们只占据嵌入空间中一个狭窄的锥形区域，而不是均匀地分布到所有方向上 [37]。

为何这对应用很重要：

- RAG / 检索：如果所有嵌入无论内容如何，余弦相似度都大于 0.7，那么检索排序将几近随机——系统无法区分相关与不相关的段落。
- 推荐系统：只有当几何结构保留了有意义的相似性结构时，使用预训练 LLM 嵌入来表示物品/用户才奏效。
- 聚类：各向异性的嵌入会压扁簇，使人无法发现自然分组。

**解决方法：白化（Whitening）。** 一个简单有效的修复手段是白化 [38]——一种使嵌入分布变为各向同性（零均值、单位协方差）的线性变换：

$$
\tilde{h} = D^{-1/2} U^\top (h - \mu)
\quad (1.4)
$$

其中 $\mu$ 是均值嵌入，$U D U^\top$ 是协方差矩阵 $\Sigma = \frac{1}{N} \sum_{i} (h_i - \mu)(h_i - \mu)^\top$ 的特征分解。

![图 1.6：嵌入空间中的各向同性与各向异性。左：各向同性嵌入均匀分布，使余弦相似度成为语义相关性的可靠度量。右：各向异性嵌入（如 BERT 中所见）聚集成一个狭窄锥体，导致所有词对无论语义内容如何都具有很高的余弦相似度。白化变换该空间以恢复各向同性。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p44-05.png)

**实践中的白化**

- 它做什么：旋转并缩放嵌入空间，使所有方向具有相等的方差（单位协方差）。
- 效果：余弦相似度变得有意义——语义相似的词对得分高，不相似的词对得分低。
- 额外好处：可同时通过只保留前 $k$ 个特征向量来降维（类似于 PCA），从而加速检索。
- 代价：需要在一个有代表性的语料库上计算协方差矩阵（一次性，$O(N \cdot d^2)$）。变换本身在推理时只是一次简单的矩阵乘法。
- 替代方法：对比微调（SimCSE）、基于流的归一化，或使用促进各向同性的正则项进行训练。

### 1.3.5 自注意力机制

自注意力（self-attention）是允许每个词元注意到序列中所有其他词元的核心操作，它基于相关性计算一个加权组合。

**缩放点积注意力（Scaled Dot-Product Attention）**

给定输入序列 $X \in \mathbb{R}^{n \times d}$，我们计算：

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad (W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k})
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
$$

其中 $M$ 是因果掩码（用于自回归模型）：若 $i \geq j$ 则 $M_{ij} = 0$，否则为 $-\infty$。

直觉：每个词元「注意」所有之前的词元，基于查询-键相似度计算它们值的加权平均。

**计算复杂度。** 朴素的注意力计算在序列长度上具有二次代价：

- 时间：$O(n^2 \cdot d)$ —— 计算 $Q K^\top$ 需要 $n^2$ 个点积，每个维度为 $d_k$。
- 内存：$O(n^2)$ —— 必须实体化完整的注意力矩阵才能应用 softmax。

对于 $d = 4096$、128K 词元的上下文，仅注意力矩阵就是 $128\text{K} \times 128\text{K} = 164$ 亿项（FP32 下 64 GB）。这种二次方扩展是长上下文 LLM 的根本瓶颈。

**表 1.4：注意力代价的扩展：为何朴素实现对长序列不可行。**

| 序列长度 | 注意力运算量 | 矩阵大小 | 实际影响 |
|---|---|---|---|
| 2K | 4M | 16 MB | 快速；可放入 SRAM |
| 8K | 64M | 256 MB | 配合 FlashAttention 尚可管理 |
| 32K | 1B | 4 GB | 需要内存高效内核 |
| 128K | 16B | 64 GB | 超出单 GPU 的 HBM |
| 1M | 1T | 4 TB | 无次二次方方法则不可行 |

**驯服注意力代价的方法。** 若干类方案用于应对这一二次方瓶颈：

1. **具备 IO 感知的精确注意力（FlashAttention [7]）**：不降低计算复杂度，而是通过在能放入 SRAM 的分块（tiles）中计算注意力，免去了在 HBM 中实体化 $n \times n$ 矩阵的需要。关键在于，FlashAttention 与下面的稀疏模式是正交的——它是一个执行引擎，而非一种注意力模式。生产系统通常将 FlashAttention 与滑动窗口或块稀疏掩码结合，兼得 IO 效率与更少的 FLOPs。我们在 1.6 节中详细讨论该算法。
2. **滑动窗口 / 局部注意力**：每个词元只注意最近的 $w$ 个词元（例如 $w = 4096$）。代价变为 $O(n \cdot w)$——在 $n$ 上线性。Mistral [26]（窗口 = 4096）和 Longformer [39] 采用此法。以牺牲全局上下文换取效率；在实践中效果良好，因为大多数注意力都是局部的。在现代技术栈中，滑动窗口掩码在 FlashAttention 内核内部执行。
3. **稀疏注意力模式**：将局部窗口与周期性的全局词元结合（例如每第 512 个词元注意到所有词元）。BigBird [40] 和 LongT5 [41] 使用此法。以 $O(n\sqrt{n})$ 的代价保留了部分长程连接。同样，FlashAttention 充当非零注意力块的底层内核。
4. **线性注意力 / 状态空间模型**：利用结合律以 $\phi(Q)(\phi(K)^\top V)$ 替换 $\text{softmax}(Q K^\top) V$，或将其重表述为一种递推（Mamba [42]、RWKV [43]）。理论上总计 $O(n \cdot d^2)$。与上面 2–3 不同，这些是架构层面的替换，会改变模型的表达能力——不含 softmax 的注意力本质上表达能力更弱，经验上这些模型在需要精确长程检索或复杂推理的任务上仍落后于 transformer。
5. **KV 缓存压缩**：推理时压缩或驱逐旧的 KV 对以限制内存。技术包括：H2O [44]（heavy-hitter oracle——只保留高注意力的键）、StreamingLLM [45]（保留初始「注意力汇聚」词元 + 近期窗口），以及量化 KV 缓存 [46]。

**FlashAttention + 稀疏模式 = 鱼与熊掌兼得**

一个常见的误解是 FlashAttention 是稀疏注意力的替代品。并非如此——它是注意力内核的一项 IO 优化，可与任意注意力掩码自由组合。现代生产系统（如 Mistral、DeepSeek）在滑动窗口或块稀疏掩码之下以 FlashAttention 作为执行引擎。这同时带来了（来自稀疏性的）更少 FLOPs 和（来自分块的）最优内存访问模式。RingAttention [47] 将其进一步扩展到多设备环境，沿序列维度把分块计算分布到各 GPU 上。

线性注意力和状态空间模型（Mamba、RWKV）是一种真正不同的架构选择——它们以 $O(n)$ 计算换取牺牲全两两交互。虽然在理论上很优雅，但它们在知识密集型或长程推理任务上尚未匹敌 transformer 的质量，前沿实验室仍继续使用精确注意力（配合 FlashAttention + 稀疏性）作为骨干。

### 1.3.6 多头注意力

与其计算单一的注意力函数，多头注意力（multi-head attention）并行运行多个注意力操作，每个学习聚焦输入的不同方面（语法、语义、位置等）。

**多头注意力**

不再使用具有 $d$ 维键/值的单一注意力函数，而是使用 $H$ 个并行头，维度 $d_k = d/H$：

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
$$

每个头可以学习不同的注意力模式（例如一个头学语法、另一个学语义、又一个学位置邻近性）。

**分组查询注意力（Grouped Query Attention, GQA）**：Llama-3 [25] 使用的 K、V 头少于 Q 头（例如 8 个 KV 头由 32 个 Q 头共享）。这使得 KV 缓存大小减少 4 倍，而质量损失极小。

### 1.3.7 位置编码

Transformer 在构造上是置换等变（permutation-equivariant）的——没有位置信息，模型无法区分「the cat sat on the mat」与「mat the on sat cat the」。位置编码注入序列顺序信号，使注意力能够对词元距离与方向进行推理。

**表 1.5：现代 LLM 中的位置编码方法。**

| 方法 | 使用者 | 核心思想 |
|---|---|---|
| 正弦（Sinusoidal） | 原始 Transformer | 不同频率下的固定 sin/cos。非学习。 |
| 学习的绝对位置（Learned Absolute） | GPT-2 [31]、BERT [27] | 每个位置学习一个嵌入。受限于训练长度。 |
| RoPE（旋转） | Llama [25]、Qwen [32]、Mistral [26] | 按依赖位置的角度旋转 $Q,K$ 向量。通过 NTK-aware scaling 进行外推。 |
| ALiBi | BLOOM [48]、MPT [49] | 无位置嵌入；给注意力分数加上线性偏置 $-m|i-j|$。简单，外推良好。 |

**正弦（固定）位置编码。** 在原始 Transformer [6] 中引入，该方法在几何间隔的频率上使用固定的正弦函数：

$$
\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \quad
\text{PE}(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

其中 pos 是词元位置，$i$ 是维度索引，$d$ 是模型维度。

动机：每个频率以不同的尺度编码位置（类似于二进制计数）。作者假设模型可以学会注意相对位置，因为 $\text{PE}(\text{pos}+k)$ 可表示为 $\text{PE}(\text{pos})$ 的线性函数。

优点：零学习参数；确定性；理论上支持任意长度。

缺点：在实践中超出训练长度后外推不佳；模型必须间接地从绝对信号中解码出相对位置；基本已被取代。

**学习的绝对位置嵌入。** 由 GPT-2 [31] 和 BERT [27] 使用：一个可学习的嵌入矩阵 $E_\text{pos} \in \mathbb{R}^{L_\text{max} \times d}$ 被加到词元嵌入上：

$$
h^{(\text{pos})}_0 = \text{TokenEmbed}(x_\text{pos}) + E_\text{pos}[\text{pos}]
$$

动机：让模型学习对任务最优的任何位置表示，而不是强加一个固定结构。

优点：最大灵活性；实现简单；在短序列上常常优于正弦编码。

缺点：硬编码的最大长度 $L_\text{max}$；超出它即无法泛化；靠近 $L_\text{max}$ 末端的嵌入训练不足；增加 $L_\text{max} \times d$ 个参数。

**旋转位置嵌入（Rotary Position Embedding, RoPE）。** RoPE [50] 通过在二维子空间中旋转查询和键向量来编码位置：

$$
\text{RoPE}(x_m, m) =
\begin{pmatrix}
x_m^{(1)} \\
x_m^{(2)} \\
\vdots \\
x_m^{(d-1)} \\
x_m^{(d)}
\end{pmatrix}
\odot
\begin{pmatrix}
\cos m\theta_1 \\
\cos m\theta_1 \\
\vdots \\
\cos m\theta_{d/2} \\
\cos m\theta_{d/2}
\end{pmatrix}
+
\begin{pmatrix}
-x_m^{(2)} \\
x_m^{(1)} \\
\vdots \\
-x_m^{(d)} \\
x_m^{(d-1)}
\end{pmatrix}
\odot
\begin{pmatrix}
\sin m\theta_1 \\
\sin m\theta_1 \\
\vdots \\
\sin m\theta_{d/2} \\
\sin m\theta_{d/2}
\end{pmatrix}
$$

其中 $\theta_i = 10000^{-2i/d}$，$m$ 是位置索引。关键性质是，旋转后的查询和键之间的点积只依赖于相对位置：

$$
\langle \text{RoPE}(q_m, m),\; \text{RoPE}(k_n, n) \rangle = f(q_m, k_n, m - n)
$$

动机：无需显式偏置项即可实现相对位置编码，同时保持与线性注意力和 KV 缓存的兼容性。

优点：天然是相对的；无额外参数；兼容高效推理；可通过 NTK-aware scaling [51] 或 YaRN（调整 $\theta$ 基或对频率插值）扩展到更长的上下文。

缺点：每次注意力运算的计算略多（旋转 + 交错）；外推需要显式的缩放策略；二维子空间中的旋转所强加的结构对某些任务可能并非最优。

**RoPE 长度扩展**

要将一个在长度 $L$ 上训练的 RoPE 模型扩展到上下文长度 $L' > L$：

- **位置插值**：按 $L/L'$ 缩放位置，使所有位置落入 $[0, L]$。简单但会压缩分辨率。
- **NTK-aware scaling**：增大 $\theta$ 基（例如 $10000 \to 10000 \cdot (L'/L)^{d/(d-2)}$），有效地拉伸高频分量而保留低频分量。
- **YaRN [51]**：把 NTK scaling 与注意力温度校正 $t = 0.1\ln(s) + 1$ 结合，以补偿更远距离处增加的熵。

**ALiBi（带线性偏置的注意力）。** ALiBi [52] 采取了一种截然不同的方式：完全没有位置嵌入。取而代之的是，从注意力分数中减去一个静态的线性惩罚：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} - m \cdot \big[|i - j|\big]_{i,j}\right) V
$$

其中 $m$ 是一个头特有的斜率（按几何方式设定：对于共 $H$ 个头中的第 $h$ 个头，$m_h = 2^{-8h/H}$）。偏置 $-m|i-j|$ 创建了一个软的局部注意力窗口，其宽度因头而异。

动机：位置应当在不干扰嵌入空间的前提下，使注意力偏向附近词元（recency prior，近因先验）。通过纯粹地在注意力分数空间中操作，ALiBi 避免了用位置信号污染词元表示。

优点：极佳的长度外推（在 1k 上训练，在 8k+ 上工作）；零参数；实现简单；头特有的斜率提供多尺度局部性。

缺点：对于需要精确长程位置推理的任务（例如「第 5 个词是什么？」）表达能力较弱；线性衰减是一个很强的归纳偏置，可能并不适合所有领域；由于 RoPE 在短上下文上表现更好，近期模型中基本已被 RoPE 取代。

**表 1.6：位置编码对比：实际权衡。**

|  | 正弦 | 学习的绝对 | RoPE | ALiBi |
|---|---|---|---|---|
| 额外参数 | 无 | $L_\text{max} \times d$ | 无 | 无 |
| 位置类型 | 绝对 | 绝对 | 相对 | 相对（隐式） |
| 长度外推 | 差 | 无 | 良好（配合缩放） | 极佳 |
| 计算开销 | 可忽略 | 可忽略 | 小 | 可忽略 |
| 主导时期 | 2017–19 | 2018–20 | 2022–至今 | 2022–23 |

**扩展到极长上下文（100K–1M+ 词元）。** 现代前沿模型（Claude [53] 200K–1M 上下文、Gemini 1.5 [54] 1M+、GPT-4 [23] 128K）要求位置编码在远超训练长度后仍保持忠实。如今的主流方案：

1. **带频率缩放的 RoPE**：把 RoPE 扩展到训练长度之外的标准做法。无需重训，而是对基频率 $\theta$ 重缩放：

$$
\theta'_i = \theta_i \cdot \left(\frac{L_\text{target}}{L_\text{train}}\right)^{2i/d}
$$

变体包括：

- **线性缩放（位置插值）[55]**：简单地把位置索引除以因子 $s$。廉价但在高扩展比下质量下降。
- **NTK-aware scaling [51]**：缩放基频率 $\theta = 10000 \to 10000 \cdot s^{d/(d-2)}$。在扩展低频（全局）范围的同时保留高频（局部）信息。
- **YaRN [51]**（Yet another RoPE extensioN）：把 NTK scaling 与注意力温度校正以及在少量长上下文语料上的微调结合。Llama-3 用它从 8K 训练扩展到 128K 部署。
- **动态 NTK [51]**：在推理时根据实际序列长度即时调整缩放因子。无需固定扩展比——模型随上下文增长而自适应。
2. **在长数据上持续预训练**：即便有 RoPE 缩放，模型仍能受益于在长文档上进行一段简短的持续预训练阶段（1–5B 词元）。这教会模型真正去使用远处的上下文，而不仅仅是位置上容忍它。Llama-3.1 采用渐进式调度：8K → 64K → 128K。
3. **Ring Attention / 块状并行 [47]**：对于超出单 GPU 内存的序列（1M+ 词元），Ring Attention 以环状拓扑把序列分布到各 GPU。每个 GPU 持有一个块并沿环传递 KV 块，计算局部注意力分块。这使得内存随 GPU 数量线性扩展，同时保持精确注意力。
4. **混合架构**：某些系统为大多数层组合局部滑动窗口（例如 4K），并在选定层（例如每第 4 层）使用全注意力。这为大多数计算提供 $O(n \cdot w)$ 的代价，同时维持全局信息流。

**长上下文 ≠ 长上下文使用**

一个具有 1M 上下文长度的模型并不一定能有效使用全部 1M 词元。「迷失在中间」（lost in the middle）现象 [56] 表明，模型倾向于聚焦长上下文的开头和结尾，而未能充分利用中间的信息。有效的长上下文利用既需要位置编码支持，又需要在奖励长程检索的任务上训练。

### 1.3.8 前馈网络（MLP）

每个 transformer 块包含一个独立地应用于每个位置的 MLP：

$$
\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2
$$

其中 $W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$。现代 LLM 使用：

- **SwiGLU 激活**：$\text{FFN}(x) = W_2(\text{Swish}(W_1 x) \odot W_3 x)$ —— 由 Llama [25]、Mistral [26] 使用。需要 3 个权重矩阵，但给出更好的性能。
- 隐藏维度通常为 $8/3 \times d$（向上取整到 256 的倍数以提升 Tensor Core 效率）。

**作为记忆的 FFN**

近期工作 [57] 表明 FFN 层充当键值记忆：$W_1$ 的行是键（要匹配的模式），$W_2$ 的列是值（要输出的信息）。FFN 基于当前隐藏状态「检索」存储的知识。

### 1.3.9 层归一化

层归一化（layer normalization）通过对特征维度上的激活做归一化来稳定训练。它相对于注意力/FFN 子层的位置显著影响训练动力学。

**LayerNorm 如何工作。** 给定一个隐藏状态向量 $x \in \mathbb{R}^d$（单个词元的表示），LayerNorm [58] 计算：

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\quad (1.5)
$$

其中：

- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$（沿 $d$ 个特征维度的均值）
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$（沿特征维度的方差）
- $\gamma, \beta \in \mathbb{R}^d$ 是学得的缩放与平移参数（逐维度）
- $\epsilon \approx 10^{-5}$ 防止除零

与 BatchNorm 的关键区别：LayerNorm 是对单个样本的特征维度做归一化，而不是跨 batch。这使其与 batch 大小无关，且在训练和推理时工作方式相同。

**RMSNorm——现代简化。** RMSNorm [59] 舍弃了去均值步骤，仅按均方根归一化：

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
\quad (1.6)
$$

没有 $\beta$（平移）参数，也没有减去均值——只有缩放。这为每个词元节省了一次归约运算，在 GPU 上约快 5–10%，同时达到同等的模型质量。所有现代 LLM（Llama、Mistral、Qwen）都使用 RMSNorm。

**Pre-LN vs Post-LN**

- **Post-LN**（原始 Transformer）：$h + \text{LayerNorm}(\text{Attn}(h))$。需要仔细的预热；训练可能不稳定。
- **Pre-LN**（GPT-2 及之后，所有现代 LLM）：$h + \text{Attn}(\text{LayerNorm}(h))$。稳定训练；允许更高的学习率。
- **RMSNorm**（Llama [25]、Mistral [26]）：无去均值的简化 LayerNorm：$\text{RMSNorm}(x) = x/\text{RMS}(x) \cdot \gamma$。略快，质量相同。

**归一化对深度网络为何重要**

没有归一化，激活往往会随层数呈指数增长或衰减（激活爆炸/消失）。一个没有 LayerNorm 的 128 层 transformer 会在第一层和最后一层之间看到量级相差 $10^{30}$ 倍。归一化把每一层的输出约束在一个可预测的范围内，从而实现稳定的梯度流，并允许优化器在整个网络中使用一致的学习率。

### 1.3.10 模型规模参考

下表总结了广泛使用的开源权重模型的关键架构参数（截至 2025 年的最新版本），为理解规模和设计选择提供快速参考。

**表 1.7：流行的开源权重 LLM 的架构参数（2024–2025 一代）。**

| 模型 | 参数量 | 层数 | $d$ | 头数 | KV 头数 | 上下文 |
|---|---|---|---|---|---|---|
| Llama-3.1 8B [25] | 8B | 32 | 4096 | 32 | 8 | 128K |
| Llama-3.1 405B [25] | 405B | 126 | 16384 | 128 | 8 | 128K |
| Llama-4 Maverick [60] | 400B（17B 激活） | 48 | 5120 | 40 | 8 | 1M |
| Mistral Large 2 [61] | 123B | 88 | 12288 | 96 | 8 | 128K |
| Qwen-2.5 72B [32] | 72B | 80 | 8192 | 64 | 8 | 128K |
| DeepSeek-V3 [62] | 671B（37B 激活） | 61 | 7168 | 128 | MLA | 128K |

注：标有「激活」参数的模型使用混合专家（Mixture of Experts, MoE）架构——总参数量表示模型容量，而激活参数量反映每词元的计算代价。DeepSeek-V3 使用多头潜在注意力（Multi-head Latent Attention, MLA）而非标准的 GQA，把 KV 压缩到一个低秩潜在空间。

### 1.3.11 注意力病态

尽管注意力机制功能强大，但它表现出一些从业者必须理解的系统性失效模式——尤其是在扩展到长上下文或解释模型行为时。

**注意力汇聚（Attention Sink）**

现象。Xiao 等人 [63] 发现 transformer 模型会把不成比例的高注意力分数分配给序列中的第一个词元——无论其语义内容如何。即便第一个词元是无意义的 `<BOS>` 标记，跨所有层的注意力头都持续地注意到它，有时占据总注意力的 20–50%。

为何发生。softmax 注意力必须产生一个合法的概率分布（$\sum_j \alpha_j = 1$）。当没有任何键对某个查询特别相关时，模型需要一个「倾倒」位置来安放未使用的注意力质量。在训练期间，第一个词元成为这种默认汇聚点，因为它始终存在且在位置上可预测。它充当一个无操作（no-op）的注意力目标——模型学会了把无关注意力路由到那里，而不是不可预测地分布它。

$$
\alpha_\text{sink} = \frac{\exp(q^\top k_0 / \sqrt{d})}{\sum_j \exp(q^\top k_j / \sqrt{d})} \gg \frac{1}{n}
\quad (\text{即便 } k_0 \text{ 在语义上无关})
$$

后果。

- 流式推理失效：当使用滑动窗口 KV 缓存时，驱逐第一个词元会导致困惑度灾难性飙升——模型失去了它的注意力汇聚点。
- 误导性的可解释性：朴素的注意力可视化会暗示第一个词元「重要」，而它其实只是数学上的产物。
- 上下文窗口浪费：汇聚词元占据一个 KV 缓存槽却不携带有用信息。

解决方案。

- **StreamingLLM [63]**：始终把前 $k$ 个词元（「注意力汇聚」）连同近期滑动窗口一起保留在 KV 缓存中。从而以有界内存实现无限长度生成。
- **按设计加入汇聚词元**：某些模型（例如 Mistral）在训练期间前置专门的汇聚词元，明确用于吸收残余注意力。
- **softmax 替代方案**：用 ReLU 注意力或 sigmoid 门控替换 softmax，在其中零注意力是可表示的，无需倾倒目标。

**注意力稀释（Attention Dilution）**

现象。随着序列长度 $n$ 增长，每个查询必须把注意力预算分摊到更多键上。每个词元的平均注意力权重按 $O(1/n)$ 下降，使模型越来越难以聚焦于少数真正相关的位置——这一问题被称为注意力稀释或注意力扩散 [56]。

「迷失在中间」效应。Liu 等人 [56] 表明 LLM 呈现 U 型检索曲线：放在长上下文开头或结尾的信息能被可靠检索，但中间的信息常被忽略。这是注意力稀释与 RoPE/ALiBi 位置偏置叠加的直接后果。

为何发生。

- **softmax 饱和**：键很多时，softmax 温度实际上下降，使分布更均匀（高熵）。
- **位置衰减**：RoPE 的相对位置编码引入了随距离的自然衰减，抑制了对既远离起点又远离终点的中间位置的注意力。
- **训练分布**：在较短序列上训练的模型发展出偏向近期上下文的注意力模式。

缓解策略。

- **显式检索**：把相关上下文放在提示的开头或结尾；使用 RAG 以避免依赖中间位置。
- **长上下文训练**：在长文档上训练，并将关键信息放置在不同位置 [64]。
- **分层注意力**：如 Mamba [65] 或 RWKV 这样的架构完全规避 $O(n^2)$ 注意力瓶颈。
- **地标词元（Landmark tokens）**：在上下文中插入可检索的标记，作为注意力的「路标」。
- **温度缩放**：某些实现把注意力 logits 乘以 $\log n$，以对抗长序列中的稀释。

**其他注意力现象**

**表 1.8：大型 transformer 中观察到的其他注意力模式。**

| 模式 | 描述 | 含义 |
|---|---|---|
| 注意力头特化 | 不同的头学到不同的角色：语法头、共指头、位置头 [66] | 并非所有头同等重要；许多可被剪枝 |
| 归纳头（Induction heads） | 实现 `[A][B]...[A] → [B]` 复制的头 [67] | 对上下文学习至关重要；在 2 层及以上的模型中出现 |
| 注意力坍缩（Attention collapse） | 深层网络中，注意力分布可能收敛（所有头注意相同位置） | 损害表达力；通过注意力多样性损失来应对 |
| 检索头 | 专门从上下文中检索事实信息的特定头 [68] | 解释了为何剪枝某些头会导致幻觉激增 |

### 1.3.12 为可解释性可视化注意力

注意力权重为洞察模型推理打开了一扇窗——但必须谨慎解读。

**注意力可视化方法**

原始注意力图。最简单的方法：为每个头和每层把 $n \times n$ 注意力矩阵 $A = \text{softmax}(Q K^\top / \sqrt{d})$ 绘制为热力图。诸如 BertViz [69] 这样的工具可渲染交互式的多头可视化。

注意力逐层累积（Attention rollout）。单层的原始注意力具有误导性，因为信息会通过跨所有层的残差连接流动。Abnar 和 Zuidema [70] 提出注意力逐层累积：把跨层的注意力矩阵相乘，以近似从输入到输出的总信息流：

$$
R^{(l)} = A^{(l)} \cdot R^{(l-1)}, \quad R^{(0)} = I
$$

其中 $A^{(l)}$ 是第 $l$ 层（在头上平均的）注意力矩阵，并经调整以包含残差连接：$A^{(l)} = 0.5 \cdot A^{(l)}_\text{raw} + 0.5 \cdot I$。

梯度加权注意力。把注意力权重与梯度信息结合，以识别哪些被注意的词元实际影响了输出 [71]：

$$
\text{Relevance}(i) = \alpha_i \cdot \left|\frac{\partial y}{\partial h_i}\right|
$$

这回应了「高注意力 ≠ 高影响」的批评（一个词元可能获得高注意力，却通过一条近零权重的路径被处理）。

**注意力不等于解释**

Jain 和 Wallace [72] 表明，注意力权重往往与基于梯度的特征重要性不相关，而且对抗性的注意力分布可以产生相同的输出。请把注意力可视化当作一个生成假设的工具，而非忠实的解释。对于因果归因，请优先使用基于梯度的方法、探测（probing）或机制可解释性（mechanistic interpretability）。

**用稀疏自编码器（SAE）做机制可解释性**

可解释性问题。transformer MLP 和残差流中的单个神经元通常是多义的（polysemantic）——单个神经元会对多个不相关的概念激活（例如「蓝色 AND 学术引用 AND 单词 'the'」）。这使得直接的神经元级解释不可靠。

稀疏自编码器（Sparse Autoencoders）。Cunningham 等人 [73] 和 Bricken 等人 [74] 证明，在模型激活上训练一个稀疏自编码器（SAE）可以把多义表示分解为单义特征（monosemantic features）——每个对应单一概念的可解释方向：

$$
h = W_\text{dec} \cdot \text{ReLU}(W_\text{enc} \cdot x + b_\text{enc}) + b_\text{dec}
$$

其中 $W_\text{enc} \in \mathbb{R}^{m \times d}$，$m \gg d$（过完备基），且 ReLU + 稀疏惩罚确保每个输入只激活少数特征。

来自 SAE 可解释性的关键发现：

- 特征是单义的：每个编码单一的人类可解释概念（「用 Python 写的代码」「提及金门大桥」「第一人称叙事」）[74]。
- 特征是可操纵的：把一个特征的激活钳制为高/低可直接控制模型行为（例如强行开启「金门大桥」特征会让模型在每次响应中都提到它）[75]。
- 特征可组合：复杂行为从简单特征的组合中涌现。
- SAE 可扩展：Templeton 等人 [75] 在 Claude 3 Sonnet 上训练了多达 34M 特征的 SAE，为安全相关的概念（欺骗、谄媚、危险请求）找到了可解释特征。

**SAE 训练配方**

1. 从一个特定模型层跨大规模语料收集激活。
2. 在隐藏层上带 L1 惩罚训练一个稀疏自编码器：$\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1$。
3. 学得的编码器方向（$W_\text{enc}$ 的行）是候选特征。
4. 验证：为每个特征找出最大激活示例，并检查语义连贯性。
5. 可选：测量特征吸收和死亡特征以评估 SAE 质量。

**自然语言自编码器（Anthropic，2026）**

尽管 SAE 把激活分解为可解释的向量，但其特征仍需人工检查最大激活示例才能理解。Anthropic 的自然语言自编码器（Natural Language Autoencoders, NLAEs）[76] 采取了根本不同的方法：它们用自然语言描述替换稀疏瓶颈，使可解释性自动化。

## 1.4 预测头:Transformer 输出什么

Transformer 主体为每个位置产生上下文化的隐藏状态 $h_t \in \mathbb{R}^d$。我们如何处理这些隐藏状态——即预测头(prediction head)——决定了任务。同一个 Transformer 骨干(backbone)只需更换预测头,就能服务于截然不同的目的。

### 1.4.1 语言建模头(预训练)

标准的 LM 头将最终的隐藏状态投影到词表的 logits,并用下一词元的交叉熵损失(cross-entropy loss)进行训练:

$$
P(x_{t+1} \mid x_{\le t}) = \mathrm{softmax}(W_\text{head} \cdot h_t + b) \quad (1.7)
$$

其中 $W_\text{head} \in \mathbb{R}^{|V| \times d}$(通常与嵌入矩阵绑定:$W_\text{head} = E^T$)。

![图 1.7:同一个 Transformer 骨干通过更换预测头即可支持不同任务。本文用到的三种头在最终投影层之下共享完全相同的架构。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p55-06.png)

**LM 头属性**

- **训练目标**:因果语言建模(为每个位置预测下一词元)
- **损失**:$\mathcal{L}_\text{LM} = -\dfrac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})$
- **标签**:每个词元既是输入(右移)又是目标(左移)
- **使用阶段**:在大型语料库(数万亿词元)上进行预训练
- **关键洞见**:模型把通用语言理解作为下一词元预测的副产品习得

### 1.4.2 条件生成头(SFT / 指令遵循)

对于监督微调(Supervised Fine-Tuning, SFT),其架构与 LM 头完全一致——同样是投影到词表 logits 的线性层。区别纯粹在于我们在哪里计算损失:

$$
\mathcal{L}_\text{SFT} = -\dfrac{1}{|y|} \sum_{t=1}^{|y|} \log P(y_t \mid x_\text{prompt}, y_{<t}) \quad (1.8)
$$

**条件头——与 LM 头的关键差异**

- **损失掩蔽(loss masking)**:只在响应(response)词元上计算损失,而不在提示/指令上计算。提示只提供上下文,不产生梯度信号。
- **条件化**:模型学会在特定的指令格式(系统提示、用户查询、工具调用)条件下生成响应。
- **格式词元**:特殊词元(`<|user|>`、`<|assistant|>`)引导模型产生结构化输出。
- **使用阶段**:在精选的指令-响应对上进行 SFT;也用于 RL 策略生成(产生动作/响应的策略头)。

**同一个头——不同的训练信号**

LM 头和 SFT 头在架构上完全相同(同一个 $W_\text{head}$)。唯一的区别是,在 SFT 期间我们对提示词元做损失掩蔽。这一细微改动就把一个通用文本预测器转变成了一个遵循指令的助手。该头学会根据条件化上下文"激活"不同的生成模式。

### 1.4.3 价值头(用于 RL 的回归)

在强化学习(Reinforcement Learning, RL,如 PPO、GRPO)中,我们需要估计某个状态有多好——这需要一个标量输出,而非词表 logits。价值头用一个简单的回归层取代 LM 投影:

$$
V(s_t) = w_\text{value}^T \cdot h_t + b \in \mathbb{R} \quad (1.9)
$$

其中 $w_\text{value} \in \mathbb{R}^d$,$b \in \mathbb{R}$。

**价值头属性**

- **输出**:单个标量(从该状态出发的期望累积奖励)
- **损失**:预测回报与实际回报之间的均方误差(MSE):$\mathcal{L}_V = \dfrac{1}{T} \sum_{t} (V(s_t) - R_t)^2$
- **架构**:线性层 $\mathbb{R}^d \to \mathbb{R}^1$(有时带一个小 MLP:$d \to 256 \to 1$)
- **骨干共享**:通常与策略共享 Transformer 主体(带一个独立的价值头),或使用一个完全独立的评判网络(critic network)
- **使用阶段**:PPO 的优势估计(GAE)、奖励模型打分

### 1.4.4 预测头选择汇总

表 1.9:本文用到的预测头及其训练场景。

| 头 | 输出 | 损失 | 阶段 | 用途 |
|---|---|---|---|---|
| LM 头 | $\mathbb{R}^{|V|}$ | 交叉熵(全部词元) | 预训练 | 从原始文本学习语言 |
| 条件头 | $\mathbb{R}^{|V|}$ | 交叉熵(仅响应) | SFT | 学习遵循指令 |
| 价值头 | $\mathbb{R}^1$ | MSE | RL(PPO) | 估计状态价值以用于优势 |
| 奖励头 | $\mathbb{R}^1$ | 成对排序 | RM 训练 | 为响应质量打分 |

**头的初始化很重要**

在向一个预训练 LM 添加价值头时,要把它初始化到接近零(小的随机权重)。如果用大值初始化,初始的价值估计会偏差极大,导致优势值巨大、PPO 更新不稳定。常见做法:把最终线性层用 $\mathcal{N}(0, 1/\sqrt{d})$ 初始化,或干脆初始化为零。

### 1.4.5 HuggingFace 实现

```python
from transformers import (
    AutoModelForCausalLM,
    # LM 头(预训练 + SFT)
    AutoModelForSequenceClassification,
    # 奖励头
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead
    # 价值头(PPO)
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

lm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = lm_model(**inputs)
next_token_logits = outputs.logits[:, -1, :]
probs = torch.softmax(next_token_logits, dim=-1)

messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]
formatted = tokenizer.apply_chat_template(messages, return_tensors="pt")
labels = formatted.clone()
prompt_len = len(tokenizer.apply_chat_template(messages[:1]))
labels[:, :prompt_len] = -100
loss = lm_model(input_ids=formatted, labels=labels).loss

value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs = tokenizer("Explain quantum computing", return_tensors="pt")
lm_logits, loss, values = value_model(**inputs, return_dict=False)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    # 单个标量输出
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs = tokenizer("Good response here", return_tensors="pt")
reward_score = reward_model(**inputs).logits
```

代码清单 1.3:用 HuggingFace 加载并使用不同的预测头。

**权重绑定(Weight Tying):LM 头 = 嵌入矩阵的转置**

大多数现代 LLM 把 LM 头权重与输入嵌入矩阵绑定:`lm_head.weight = model.embed_tokens.weight`。这意味着 LM 头并非一个单独学习的层——它复用了嵌入表。好处是:更少的参数(节省 $|V| \times d$)、更好的泛化能力,并且嵌入空间的几何结构直接决定了词元概率。你可以在 HuggingFace 中验证:`model.lm_head.weight is model.model.embed_tokens.weight` 对大多数模型都返回 `True`。

## 1.5 LLM 训练的优化理论

训练一个大型语言模型意味着找到一组参数 $\theta$(数十亿个权重),使得损失函数 $\mathcal{L}(\theta)$ 最小化——通常就是下一词元的负对数似然。这是一个在极高维空间中的优化问题,用于在该空间中导航的算法决定了训练是成功、发散(diverge)还是停滞(stall)。

### 1.5.1 梯度下降:基础

**什么是梯度?** 梯度 $\nabla_\theta \mathcal{L}$ 是一个指向损失上升最快方向的向量。每个分量 $\dfrac{\partial \mathcal{L}}{\partial \theta_i}$ 告诉我们,如果稍稍增大参数 $\theta_i$,损失会变化多少。要降低损失,我们沿相反方向移动:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t) \quad (1.10)
$$

其中 $\eta > 0$ 是学习率(learning rate)——即步长。这就是梯度下降 [77]。

![图 1.8:梯度下降:从随机初始化 $\theta_0$ 出发,每一步都把参数沿降低损失的方向移动,步长由学习率 $\eta$ 控制。该过程收敛到(局部)最小值。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p58-07.png)

**为什么完整梯度下降不切实际。** 计算精确梯度需要在整个训练数据集(LLM 为数万亿词元)上评估损失。这在计算上是不可行的——单次梯度步就需要对全部数据做一次完整遍历。

**随机梯度下降(Stochastic Gradient Descent, SGD)。** 解决方案:从数据的一个小型随机子集(mini-batch)估计梯度 [78]:

$$
\nabla_\theta \mathcal{L}(\theta) \approx \dfrac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(\theta; x_i)
$$

其中 $B$ 是批大小(LLM 通常为 1K–4M 词元)。mini-batch 梯度是真实梯度的一个有噪声但无偏的估计。

**为什么 mini-batch SGD 有效**

- **计算效率**:每一步的代价是 $O(B)$ 而非 $O(N_\text{total})$。当 $B = 4096$ 词元、总计 $15T$ 词元时,每一步便宜了约 40 亿倍。
- **噪声即正则化**:随机噪声有助于逃出尖锐的局部极小值,找到泛化更好的更平坦区域。
- **GPU 利用率**:mini-batch 足够大,可以充分利用 GPU 并行(矩阵乘法变成计算受限而非访存受限)。
- **收敛性**:理论上以 $O(1/\sqrt{T})$ 的速率收敛到局部极小值(比精确 GD 的 $O(1/T)$ 慢,但每一步便宜数百万倍)。

**从 SGD 到自适应方法。** 带动量的 SGD 在视觉模型(CNN)上效果很好,但 LLM 训练需要自适应优化器——即维护每个参数各自学习率的算法。

### 1.5.2 为什么朴素的 SGD 对 LLM 失效

随机梯度下降按下式更新权重:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

**SGD 用于 LLM 的问题**

- **每层梯度尺度不同**:Transformer 早期层的梯度远小于后期层(梯度消失)。单一的学习率 $\eta$ 对某些参数太大,对另一些又太小。
- **稀疏梯度**:嵌入层只为当前 batch 中出现的词元接收梯度。大多数嵌入行的梯度为零。带动量的 SGD 在零梯度行上浪费动量。
- **鞍点(saddle points)**:高维损失曲面有许多鞍点。SGD 可能停滞;自适应方法能更快逃离。
- **对学习率敏感**:SGD 需要仔细调参;$\eta$ 变化 2 倍就可能引起发散。

### 1.5.3 Adam——自适应矩估计

Adam [79] 维护每个参数对一阶矩(梯度均值)和二阶矩(梯度的未中心化方差)的估计。

**Adam 更新公式**

给定梯度 $g_t = \nabla_\theta \mathcal{L}(\theta_t)$、超参数 $\beta_1, \beta_2, \epsilon, \eta$:

**步骤 1——更新有偏一阶矩估计:**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

**步骤 2——更新有偏二阶矩估计:**

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**步骤 3——偏差修正:**

$$
\hat{m}_t = \dfrac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \dfrac{v_t}{1 - \beta_2^t}
$$

**步骤 4——参数更新:**

$$
\theta_{t+1} = \theta_t - \eta \cdot \dfrac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

典型取值:$\beta_1 = 0.9$,$\beta_2 = 0.95$ 或 $0.999$,$\epsilon = 10^{-8}$,$\eta = 10^{-4}$ 到 $10^{-5}$。

**各项的作用**

- $m_t$(动量):梯度的指数移动平均(EMA)。平滑掉有噪声的梯度估计。$\beta_1 = 0.9$ 意味着当前梯度贡献 10%,历史贡献 90%。
- $v_t$(自适应学习率):平方梯度的 EMA。梯度持续较大的参数会得到更小的有效学习率($\eta/\sqrt{v_t}$),梯度较小的参数则得到更大的有效学习率。这是处理各层梯度尺度不同的关键。
- $\hat{m}_t, \hat{v}_t$(偏差修正):当 $t = 1$ 时,$m_1 = (1 - \beta_1) g_1$ 比真实均值小得多。除以 $(1 - \beta_1^t)$ 修正了这一初始化偏差。否则早期步会太小。
- $\epsilon$(数值稳定性):防止除零。同时也充当有效学习率的下限。

### 1.5.4 AdamW——解耦的权重衰减

AdamW [80] 修复了一个细微但重要的问题:权重衰减(weight decay)与自适应优化器交互的方式。

**为什么在 Adam 中 L2 正则化 $\ne$ 权重衰减**

带 L2 正则化时,损失变为 $\mathcal{L} + \dfrac{\lambda}{2} \|\theta\|^2$,因此梯度为 $g_t + \lambda \theta_t$。在 Adam 中,这个正则化梯度被自适应因子 $1/\sqrt{\hat{v}_t}$ 缩放:

$$
\theta_{t+1} = \theta_t - \eta \cdot \dfrac{\hat{m}_t + \lambda \theta_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

$\hat{v}_t$ 大(梯度方差大)的参数得到的正则化更少。这不是我们想要的——权重衰减应当是均匀的。

**AdamW——解耦的权重衰减**

AdamW(Loshchilov & Hutter, 2017)把权重衰减直接作用于参数,置于自适应缩放之外:

$$
\theta_{t+1} = \theta_t - \eta \cdot \dfrac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t
$$

权重衰减项 $\eta \lambda \theta_t$ 不被 $\sqrt{\hat{v}_t}$ 除。这给出了对所有参数都均匀的正则化,无论其梯度历史如何。

典型取值:LLM 训练中 $\lambda = 0.1$。

**LLM 永远用 AdamW——绝不用朴素 Adam**

Adam 与 AdamW 的差别细微但很重要。使用 Adam + L2 时,有效权重衰减对梯度方差小的参数(如偏置、LayerNorm 参数)更强,对梯度方差大的参数(如注意力权重)更弱。AdamW 给出预期的均匀正则化。大多数框架默认使用 AdamW;请务必检查你的优化器类。

### 1.5.5 学习率——最重要的超参数

表:各训练阶段的典型学习率。

| 阶段 | 典型 LR | 说明 |
|---|---|---|
| 预训练(从零开始) | 1e-4 至 3e-4 | 大模型、大 batch |
| 持续预训练 | 1e-5 至 1e-4 | 较小的 LR 以保留已有知识 |
| SFT(监督微调) | 1e-5 至 2e-5 | 标准区间 |
| LoRA 微调 | 1e-4 至 3e-4 | 适配器权重用更高的 LR |

RL 学习率(PPO、DPO、GRPO)参见 §11.15。

### 1.5.6 学习率预热(Warmup)

**为什么需要预热**

训练开始时,$v_t$(二阶矩估计)被初始化为零。经过偏差修正:$\hat{v}_t = v_t / (1 - \beta_2^t)$。当 $t = 1$、$\beta_2 = 0.999$ 时:$\hat{v}_1 = v_1 / (1 - 0.999) = 1000 v_1$。这意味着有效学习率为 $\eta / \sqrt{1000 v_1}$——远小于预期。

然而,如果首个梯度异常大(初始化时常见),二阶矩估计可能被这个离群值主导,导致早期步混乱。预热通过从一个很小的 LR 开始并逐步增大来缓解这一问题,给 $v_t$ 时间去累积一个可靠的估计。

- **线性预热**:$\eta_t = \eta_\text{max} \times t / T_\text{warmup}$
- **典型预热时长**:预训练占总步数的 1–5%;微调为 3–10%(较短的训练需要成比例地更多预热)
- **对于 SFT**:50–200 个预热步较为典型

### 1.5.7 学习率调度

![图 1.9:常见的学习率调度。所有调度都包含线性预热阶段。WSD(Warmup-Stable-Decay,预热-稳定-衰减)是新兴的预训练标准。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p61-08.png)

**(a) 常数(Constant)。** 最简单的调度。适合短微调训练——不希望过度衰减 LR。风险:没有退火意味着模型可能无法收敛到最尖锐的极小值。

**(b) 余弦衰减(Cosine Decay)。**

$$
\eta_t = \eta_\text{min} + \dfrac{1}{2}(\eta_\text{max} - \eta_\text{min}) \left(1 + \cos\left(\dfrac{t - T_\text{warmup}}{T - T_\text{warmup}} \pi\right)\right)
$$

预训练和 SFT 的标准。平滑衰减避免了 LR 的突变。$\eta_\text{min}$ 通常取 $\eta_\text{max}/10$。

**(c) 线性衰减(Linear Decay)。** 比余弦更简单,经验结果相近。在希望任一步可预测 LR 时更受青睐。

**(d) WSD——Warmup-Stable-Decay。** 大规模预训练的新标准 [25, 81]。分三个阶段:

1. **预热(Warmup)**:线性爬升至 $\eta_\text{max}$(占步数的 1–5%)
2. **稳定(Stable)**:在大部分训练期间保持常数 $\eta_\text{max}$
3. **衰减(Decay)**:快速以余弦或线性衰减到 $\eta_\text{min}$(最后 10–20% 的步数)

关键优势:稳定阶段允许在任意时刻打检查点并继续训练。衰减阶段可在任何一次训练的末尾施加。

**(e) 带重启的余弦(SGDR)。** 周期性重启把 LR 重置到 $\eta_\text{max}$。有助于逃出局部极小值。在 LLM 中较少见;对小模型更有用。

### 1.5.8 梯度裁剪

**梯度裁剪(Gradient Clipping)**

如果梯度的全局范数超过阈值,梯度裁剪就对其重新缩放:

$$
g_t \leftarrow g_t \cdot \min\left(1, \dfrac{\tau}{\|g_t\|_2}\right)
$$

其中 $\tau$ 是 `max_grad_norm`(通常为 1.0)。

**梯度裁剪 vs. 降低学习率**

梯度裁剪和降低学习率都能限制参数更新的幅度。区别在于:裁剪保留了梯度的方向(只缩放幅度),而更小的 LR 均匀缩放所有更新。裁剪更适合处理偶发的大梯度,而不拖慢正常的训练步。

**综合起来:HuggingFace 优化器配置**

下面的代码片段展示了本节的概念——带解耦权重衰减的 AdamW(§1.6.6)、带线性预热的余弦学习率调度(§1.6.7)和梯度裁剪(§1.6.8)——如何使用 HuggingFace transformers 库在实践中组合到一起。

```python
from transformers import TrainingArguments, Trainer
from transformers import get_cosine_schedule_with_warmup
import torch

training_args = TrainingArguments(
    output_dir="./checkpoints",
    # AdamW 优化器(解耦权重衰减,§1.6.6)
    optim="adamw_torch",
    learning_rate=2e-5,
    # 预热后的峰值 LR
    adam_beta1=0.9,
    # 一阶矩衰减
    adam_beta2=0.999,
    # 二阶矩衰减
    adam_epsilon=1e-8,
    # 数值稳定性
    weight_decay=0.01,
    # 解耦的 L2 惩罚
    # 学习率调度(§1.6.7)
    lr_scheduler_type="cosine",
    # 余弦衰减到 0
    warmup_ratio=0.1,
    # 10% 的步数 = 线性预热
    # 梯度裁剪(§1.6.8)
    max_grad_norm=1.0,
    # 按全局 L2 范数裁剪
    # 混合精度(§1.6.9)
    bf16=True,
    # 在 Ampere+ GPU 上使用 BFloat16
    # 训练时长
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    # 有效 batch = 8*4 = 32
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

from torch.optim import AdamW
no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
param_groups = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(param_groups, lr=2e-5, betas=(0.9, 0.999))
total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    # 在优化器步之前裁剪梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

代码清单 1.4:组合 AdamW、余弦调度与梯度裁剪的完整优化器配置。

**实用技巧**

- **权重衰减排除**:偏置项和层归一化权重不应被正则化——它们参数量很少,对它们做正则化会损害性能 [80]。
- **预热比例**:总步数的 5–10% 是标准;预热太少配上高 LR 会让早期训练不稳定。
- **梯度累积**:在显存有限时模拟更大的 batch;裁剪作用于累积后的梯度。
- **BF16 vs. FP16**:在 Ampere+ GPU 上优先用 `bf16=True`(更宽的动态范围,免去损失缩放);在较老硬件上回退到 `fp16=True`。

### 1.5.9 混合精度训练

表:BF16 vs. FP16。

| 格式 | 指数位 | 尾数位 | 动态范围 |
|---|---|---|---|
| FP32 | 8 | 23 | $\sim 10^{-38}$ 至 $10^{38}$ |
| BF16 | 8 | 7 | 与 FP32 相同(指数相同) |
| FP16 | 5 | 10 | $\sim 6 \times 10^{-5}$ 至 65504 |

**BF16 相对 FP16 的优势:为何在 LLM 训练中范围胜过精度**

BF16 与 FP32 的指数范围相同,因此能表示同样范围的值(只是尾数精度较低)。FP16 的动态范围小得多——超过 65504 的梯度或激活会溢出(产生 NaN/Inf)。这就是为什么 FP16 训练需要损失缩放(loss scaling,把损失乘以一个大常数以使梯度保持在 FP16 范围内),而 BF16 训练通常不需要。A100 和 H100 原生支持 BF16;除非有特定理由用 FP16,否则就用 BF16。

**损失缩放(仅 FP16)。**

1. 把损失乘以缩放因子 $S$(如 $S = 2^{15}$)
2. 在 FP16 中计算梯度(被 $S$ 缩放)
3. 在优化器步之前,把梯度除以 $S$
4. 检查是否溢出(NaN/Inf);若发现,跳过该步并减小 $S$
5. 若连续 $N$ 步无溢出,则增大 $S$

**FP32 主权重(FP32 Master Weights)。** 在混合精度训练中,权重以 FP32 存储(主副本),并在前向/反向传播时转换为 BF16/FP16。优化器步在 FP32 中进行。这很重要,因为:

- 小的梯度更新($\Delta\theta \ll \theta$)在 BF16 精度下会丢失(7 位尾数约 0.8% 的相对精度)
- FP32 主权重确保许多步的小更新被精确累加
- 内存代价:2 倍的权重存储(FP32 + BF16 副本)

**FP32 主权重何时至关重要**

FP32 主权重在以下情形最为重要:

- 长时间训练运行(许多小梯度步累积)
- 小学习率(更新相对权重幅度而言很微小)

对于带大 LR 的短 SFT 运行,仅用 BF16(不用 FP32 主权重)通常也能正常工作并节省显存。对于 RL 训练,FP32 主权重是必不可少的——参见 §11.15。

**混合精度实践:HuggingFace**

```python
from transformers import TrainingArguments

args_bf16 = TrainingArguments(
    output_dir="./out",
    bf16=True,
    # BF16 前向/反向;FP32 主权重
    bf16_full_eval=True,
    # 评估时也用 BF16
    # 无需损失缩放 —— BF16 拥有与 FP32 等效的范围
)
args_fp16 = TrainingArguments(
    output_dir="./out",
    fp16=True,
    # FP16 前向/反向
    fp16_full_eval=False,
    # 评估保持在 FP32 以保证精度
    # 损失缩放由 PyTorch GradScaler 自动处理
)

import torch
use_fp16 = not torch.cuda.is_bf16_supported()
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
dtype = torch.float16 if use_fp16 else torch.bfloat16
for batch in train_dataloader:
    optimizer.zero_grad()
    # Autocast:以降低的精度运行前向传播
    with torch.autocast("cuda", dtype=dtype):
        outputs = model(**batch)
        loss = outputs.loss
    if use_fp16:
        # FP16 路径:缩放损失以防梯度下溢
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # 裁剪之前先反缩放
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        # 溢出时跳过该步
        scaler.update()
        # 调整缩放因子
    else:
        # BF16 路径:无需缩放
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```

代码清单 1.5:使用 HuggingFace 和手动 PyTorch AMP 的混合精度训练。

**关键差异:代码中的 BF16 vs. FP16**

- **BF16**:只需用 `autocast(dtype=torch.bfloat16)` 包裹——无需 scaler。代码更简单,数值更稳定。
- **FP16**:需要 `GradScaler` 防止梯度下溢。scaler 动态调整乘数;若检测到溢出(NaN),则跳过优化器步并降低缩放。
- **梯度裁剪 + FP16**:必须在 `clip_grad_norm_` 之前调用 `scaler.unscale_(optimizer)`,否则你裁剪的是被缩放后的梯度(阈值错误)。
- **显存节省**:激活内存减少(激活以 16 位存储);权重内存取决于是否保留 FP32 主副本。

### 1.5.10 各训练阶段的实用优化器设置

表:优化器超参数参考。

| 阶段 | 优化器 | LR | WD | 预热 | 调度 |
|---|---|---|---|---|---|
| 预训练 | AdamW | 3e-4 | 0.1 | 2000 步 | WSD 或余弦 |
| SFT | AdamW | 2e-5 | 0.01 | 100 步 | 余弦 |
| LoRA SFT | AdamW | 2e-4 | 0.01 | 100 步 | 余弦 |

所有阶段通用:$\beta_1=0.9$、$\beta_2=0.95$、$\epsilon=10^{-8}$、`max_grad_norm=1.0`、BF16。RL 设置参见 §11.15。

**诊断训练不稳定**

```python
import torch

def log_optimizer_stats(model, optimizer, step):
    # 梯度范数(裁剪之前)
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    # Adam 二阶矩统计(自适应 LR 的代理)
    v_norms = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'exp_avg_sq' in state:
                v_norms.append(state['exp_avg_sq'].mean().item())
    print(f"Step {step}: grad_norm={total_norm:.3f}, "
          f"mean_v={sum(v_norms)/len(v_norms):.6f}")

```

**学习率是最重要的超参数**

在实践中,把学习率调对比任何其他超参数都重要。LLM 微调的经验法则:

- 从上表中的取值开始
- 若损失发散(初次下降后又上升):LR 太高
- 若损失下降非常缓慢且早早进入平台:LR 太低
- 若损失不稳定(震荡):LR 太高或预热太短

第二重要的超参数是批大小(通过线性缩放规则影响梯度噪声和有效 LR)。其余都是次要的。

## 1.6 Flash Attention——算法与硬件感知

Flash Attention [7, 82] 是自 Transformer 以来深度学习中最具影响力的算法创新之一。它不改变注意力的数学结果——它计算出完全相同的输出——但它重构了访存模式,使得 GPU 上有限的高速 SRAM 承担全部重活,把 HBM(高带宽显存)占用从 $O(n^2)$ 降到 $O(n)$,在典型工作负载上带来 2–4 倍的端到端墙钟(wall-clock)提升。

### 1.6.1 标准注意力的内存问题

标准的缩放点积注意力(scaled dot-product attention)为:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\dfrac{QK^T}{\sqrt{d_k}}\right) V
$$

**标准注意力的内存复杂度**

对于序列长度 $n$、头维度 $d$:

- $Q, K, V \in \mathbb{R}^{n \times d}$:$O(nd)$ 内存
- $S = QK^T \in \mathbb{R}^{n \times n}$:$O(n^2)$ 内存——瓶颈所在
- $P = \text{softmax}(S) \in \mathbb{R}^{n \times n}$:又一个 $O(n^2)$
- $O = PV \in \mathbb{R}^{n \times d}$:$O(nd)$

当 $n = 8192$、$d = 128$、BF16 时:仅注意力矩阵一项就是 $8192^2 \times 2 \approx 134$ MB 每头。32 个头时,单层的注意力分数就占 4.3 GB。

**为什么 $O(n^2)$ 是灾难性的**

注意力矩阵必须被写入 HBM(对于长序列它放不进 SRAM),再读回做 softmax,然后再次读回做 PV 乘积。每一次 HBM 往返都很慢。对于 $n = 32768$(32K 上下文),注意力矩阵为 $32768^2 \times 2 \approx 2$ GB 每头——完全无法存储。

### 1.6.2 Flash Attention 的关键洞见——分块与在线 softmax

核心洞见是:我们从不需要一次性在内存中持有完整的 $n \times n$ 矩阵。只要使用在线 softmax(online softmax)技巧,我们就可以逐块地计算出输出 $O$。

**在线 softmax。** 回顾一下,softmax 出于数值稳定性需要一个全局最大值:

$$
\text{softmax}(x_i) = \dfrac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

技巧在于:我们可以在处理新块时更新运行中的最大值和归一化因子,而无需物化整行。

**在线 softmax 更新规则**

给定运行状态 $(m_\text{old}, \ell_\text{old}, O_\text{old})$ 以及一个新分数块 $s_\text{new}$:

1. $m_\text{new} = \max(m_\text{old}, \max(s_\text{new}))$
2. $\ell_\text{new} = e^{m_\text{old} - m_\text{new}} \cdot \ell_\text{old} + \sum_j e^{s_{\text{new},j} - m_\text{new}}$
3. $O_\text{new} = \dfrac{1}{\ell_\text{new}} \left( e^{m_\text{old} - m_\text{new}} \cdot \ell_\text{old} \cdot O_\text{old} + e^{s_\text{new} - m_\text{new}} \cdot V_\text{new} \right)$

这在数学上等价于一次性对所有块计算 softmax。

### 1.6.3 Flash Attention 算法

**Flash Attention 前向传播——分块(Tiling)**

设置:SRAM 大小 $M$。块大小 $B_r = \lceil M/(4d) \rceil$,$B_c = \min(\lceil M/(4d) \rceil, d)$。

1. 把 $Q$ 切分为 $T_r = \lceil n/B_r \rceil$ 个块 $Q_1, \dots, Q_{T_r}$
2. 把 $K, V$ 切分为 $T_c = \lceil n/B_c \rceil$ 个块 $K_1, \dots, K_{T_c}$
3. 初始化输出 $O \in \mathbb{R}^{n \times d}$、运行最大值 $m \in \mathbb{R}^n$、运行求和 $\ell \in \mathbb{R}^n$(全部在 HBM 中)
4. 对 $j = 1, \dots, T_c$ 的外层循环:
   - (a) 从 HBM 把 $K_j, V_j$ 加载到 SRAM
   - (b) 对 $i = 1, \dots, T_r$ 的内层循环:
     - i. 从 HBM 把 $Q_i, O_i, m_i, \ell_i$ 加载到 SRAM
     - ii. 计算 $S_{ij} = Q_i K_j^T / \sqrt{d}$(留在 SRAM 中)
     - iii. 应用在线 softmax 更新,得到新的 $m_i, \ell_i, O_i$
     - iv. 把 $O_i, m_i, \ell_i$ 写回 HBM
5. 返回 $O$

关键点:$S_{ij}$(注意力分块)在 SRAM 中计算并丢弃,从不写入 HBM。

**Flash Attention 复杂度**

| | 标准注意力 | Flash Attention |
|---|---|---|
| 内存(HBM) | $O(n^2)$ | $O(n)$ |
| HBM 读/写 | $O(n^2 d)$ | $O(n^2 d / M)$ |
| FLOPs | $O(n^2 d)$ | $O(n^2 d)$(相同) |
| 加速比 | 1× | 2–4× |

在前向传播中,总 FLOPs 仍是 $O(n^2 d)$——与标准注意力相同。Flash Attention 完全靠削减缓慢的 HBM 流量来获得速度,而非减少算术运算。(反向传播由于重计算实际上执行更多 FLOPs,但墙钟时间仍然更低,因为节省的访存带宽占主导。)

### 1.6.4 Flash Attention 2——更好的并行性

Flash Attention 2 [82] 做了三项关键改进:

1. **减少非矩阵乘法 FLOPs**:原始 FA 在内层循环中存在不必要的重缩放操作。FA2 重构了循环以最小化这些操作。在 A100 上,Tensor Core 矩阵乘法比标量运算快约 16 倍,因此内层循环中哪怕一小部分非 matmul 工作也会成为延迟瓶颈。
2. **沿序列维度更好的并行性**:FA1 只在 batch 和头上做并行。FA2 还沿查询序列维度并行,使得对长序列、小 batch 的场景有更好的 GPU 利用率。
3. **因果掩蔽优化**:对于自回归(因果)注意力,约有一半的分块被完全掩蔽。FA2 完全跳过这些块,对因果注意力相比双向注意力带来约 2 倍加速。

### 1.6.5 Flash Attention 3——Hopper 架构

Flash Attention 3 [83] 专为 H100 设计,利用了三个 Hopper 专属特性:

- **TMA(Tensor Memory Accelerator,张量内存加速器)**:H100 有一个专用硬件单元,用于在 HBM 与 SRAM 之间进行异步批量数据搬运。FA3 用 TMA 把数据加载与计算重叠,隐藏访存延迟。
- **线程束专门化(warp-specialization)**:FA3 把不同 warp 分配给不同角色(生产者 warp 通过 TMA 加载数据;消费者 warp 计算 MMA)。这是一种软件流水线技术,让访存系统和 Tensor Core 同时保持忙碌。
- **FP8 支持**:H100 支持 FP8(E4M3/E5M2)的 Tensor Core 运算,吞吐为 BF16 的 2 倍。FA3 支持带逐块量化以维持精度的 FP8 注意力。

FA3 在 FP16 注意力上达到 H100 理论峰值的 75%,而 FA2 约为 35%。

### 1.6.6 Flash Attention 4——Blackwell 架构

Flash Attention 4 [84] 面向 NVIDIA 的 Blackwell GPU(B200/GB200),后者把 Tensor Core 吞吐翻倍到 2.25 PFLOP/s(BF16),而非 matmul 单元(指数、共享内存带宽)则以较慢速率扩展。这种不对称的硬件扩展意味着瓶颈发生转移:在 Blackwell 上,注意力不再受 matmul 限制,而是受 softmax 指数运算及其周边的共享内存流量限制。

FA4 用四项关键技术应对:

- **全异步 MMA 流水线**:Blackwell 的 MMA 指令是全异步的(不同于 Hopper 的 wgmma 仍会阻塞等待完成)。FA4 重新设计流水线,在更大的分块尺寸下把 MMA、TMA 加载和 softmax 重缩放重叠,使所有硬件单元都饱和。
- **软件模拟指数运算**:不调用硬件 `ex2` 单元(它是吞吐瓶颈),FA4 在快得多的 Tensor Core 上用多项式近似模拟 $e^x$。这用额外的 matmul 指令换取指数单元的停顿。
- **条件性 softmax 重缩放**:标准 FlashAttention 每个分块都重缩放运行最大值。当新分块的最大值不超过运行最大值时(实践中常见),FA4 跳过重缩放,既节省寄存器搬运也省去同步屏障。
- **Tensor Memory + 2-CTA MMA 模式(反向传播)**:反向传播使用 Blackwell 的 Tensor Memory(一种比共享内存更大的每 SM 暂存区)和一种 2-CTA 协作模式,把 $dQ$ 累积融合到两个线程块集群上,使共享内存往返减半。

**FA4 实现:CuTe-DSL**

FA4 是首个用 CuTe-DSL 编写的 FlashAttention 版本。CuTe-DSL 是一种嵌入 Python 的 GPU 内核领域专用语言(CUTLASS 4.x 的一部分)。CuTe-DSL 的编译速度比 C++ CUTLASS 模板快 20–30 倍,同时保留对寄存器分配和流水线调度的完全控制。这极大缩短了内核开发的迭代时间。

**结果。** 在 B200 上、BF16、头维度 128(因果、序列长度 8K):

- 1613 TFLOP/s——Blackwell 峰值利用率的 71%
- 比 cuDNN 9.13(NVIDIA 的专有融合内核)快 1.3 倍
- 在同一硬件上比 Triton 快 2.7 倍

**硬件-软件协同演进**

FlashAttention 系列阐明了一个关键原则:每一代 GPU 都会转移瓶颈,要求新的算法思路,而不仅是重新编译。A80 → 访存带宽受限(FA1/FA2:分块 + 重计算)。H100 → 数据搬运受限(FA3:TMA + 线程束专门化)。B200 → 非 matmul 计算受限(FA4:软件模拟 exp + 条件性重缩放)。理解硬件瓶颈所在,是编写高效内核的前提。

## 1.7 预训练:最佳实践

预训练是 LLM 开发中最昂贵的阶段——消耗数百万 GPU 小时,需要对数据、算力和超参数进行精心编排。本节提炼了来自 Llama-3 [25]、Chinchilla [85] 和 GPT-4 [23] 的关键经验。

### 1.7.1 训练目标

所有现代的 decoder-only LLM 都使用因果语言建模(causal language modeling, CLM):

$$
\mathcal{L}_\text{CLM} = -\dfrac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

这一简单目标——配上足够的数据和规模——无需显式监督就能产生涌现能力(上下文学习、推理、指令遵循)[21]。

### 1.7.2 数据流水线

**预训练数据配方**

- **规模**:前沿模型用 1–15 万亿词元(Llama-3:15T 词元)
- **来源**:网页爬取(80%)、代码(10%)、书籍/论文(5%)、精选(5%)
- **去重**:MinHash + 精确子串去重可减少记忆(memorization)[86]

## 1.8 监督微调（SFT）

SFT 通过在精心挑选的「提示–响应」(prompt–response) 配对上训练，将一个预训练语言模型转化为能遵循指令的助手。这是原始语言建模与 RLHF 之间的桥梁。

### 1.8.1 SFT 目标

损失函数与因果语言建模(CLM)相同，但仅在响应 token 上计算：

$$
\mathcal{L}_{SFT} = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x_{\text{prompt}}, y_{<t})
$$

提示 token 提供上下文但不接收梯度（其标签设为 $-100$）。

### 1.8.2 数据质量：LIMA 原则

Zhou 等人 [87] 证明，1000 条精心筛选的示例即可匹敌在 5 万条以上含噪示例上训练的模型。关键要求如下：

- **多样性(Diversity)**：覆盖问答、摘要、代码、数学、创意写作、多轮对话
- **正确性(Correctness)**：每条响应都必须事实准确、格式良好
- **长度平衡(Length balance)**：混合短回复（单句）与长回复（多段落）
- **去污染(Decontamination)**：剔除与评测基准重叠的内容

### 1.8.3 训练配置

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./sft_output",
    max_seq_length=4096,
    packing=True,              # 将短样本打包成完整序列
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    bf16=True,
    gradient_checkpointing=True,
)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### 1.8.4 高效训练方案

标准的 HuggingFace 训练会留下相当大的性能空间。多个库可为 SFT 工作负载提供「即插即用」式的效率提升：

**Liger Kernel [88]。** 来自 LinkedIn 的一套开源 Triton 融合内核，用于在训练期间替换标准的 PyTorch 算子。主要融合包括：

- **融合交叉熵(Fused Cross-Entropy)**：将最终线性投影、softmax 和损失计算合并为单一内核——避免实例化完整的 (batch × seq × vocab) logit 张量。
- **融合 RMSNorm / SwiGLU / RoPE**：消除 LLM 常见构件的中间内存分配。
- **分块操作(Chunked operations)**：以瓦片(tile)方式处理大张量，使峰值内存保持有界。

效果：仅一行集成（`apply_liger_kernel_to_llama()`）即可获得 20% 的更高吞吐量与最高 60% 的内存降低。兼容 FSDP、DeepSpeed 和 LoRA。

**Unsloth [89]。** 一个专门的微调库，将自定义 CUDA/Triton 内核与激进的内存优化相结合：

- 对 LoRA 层进行手动反向传播（避免自动求导开销）。
- 4-bit QLoRA 配合融合反量化——可在单张 48 GB GPU 上训练 70B 模型。
- 针对各架构（Llama、Mistral、Qwen、Gemma）专用的智能 RoPE 与注意力内核融合。

效果：比原生 HuggingFace + PEFT 快 2–5 倍，VRAM 占用减少 60–70%。对单 GPU 和消费级硬件工作流尤为有效。

**torchtune [90]。** Meta 的原生 PyTorch 微调库（开发于 2025 年放缓），围绕可组合性而非单体抽象设计：

- 纯 PyTorch——没有 trainer 类；配方是可读的单文件脚本。
- 原生集成 `torch.compile`、FSDP2 和激活检查点。
- 一等支持 QLoRA、全量微调和知识蒸馏。
- 内置感知量化的训练(QAT)，用于训练后压缩。

效果：速度可与自定义方案媲美，同时具备完整的可调试性且无框架锁定。

**选择效率技术栈**

- 单 GPU（≤1）上快速 LoRA/QLoRA：Unsloth（最快训练到完成时间，最少配置）
- 多 GPU 全量微调：TRL/DeepSpeed + Liger Kernel（大规模下最佳吞吐）
- 研究 / 自定义训练循环：torchtune（透明、可改造、原生 PyTorch）

这些方案是互补的：Liger 内核可在 TRL 和 torchtune 工作流中同时使用。

### 1.8.5 最佳实践

表 1.11：SFT 训练准则。

| 实践 | 详情 |
|---|---|
| 打包(Packing) | 将多个短样本拼接为一个序列（以 EOS 分隔）。避免 padding 浪费。 |
| NEFTune [91] | 给嵌入添加均匀噪声（$\alpha = 5$）。以零成本将 MT-Bench 提升 5–15%。 |
| 聊天模板(Chat template) | 始终使用模型的原生模板。模板不匹配会降低质量。 |
| 训练轮数(Epochs) | 大数据集 2–3 轮；小型（<1 万）精选集最多 5 轮。过训练会导致格式记忆。 |

**仅有 SFT 是不够的**

SFT 教会模型格式和基本的指令遵循，但无法可靠地教会：偏好（哪个响应更好——需要 RLHF/DPO）、拒答（何时不应回答——需要安全训练）、校准（说出「我不知道」——需要以真实性为奖励的 RL），或复杂推理（多步链条——需要以可验证奖励的 RL）。完整流水线是：预训练(Pretrain) → SFT → RLHF/DPO。

## 1.9 LoRA 与参数高效微调

对一个 70B 模型进行全量微调，需要存储 700 亿个可训练参数及其优化器状态（560+ GB 内存）。LoRA [92]（Low-Rank Adaptation，低秩自适应）提供了一种仅用不到 1% 的参数即可完成微调、同时达到相近质量的方法。

### 1.9.1 LoRA 的核心洞察

**LoRA 核心思想**

不更新完整的权重矩阵 $W \in \mathbb{R}^{d \times d}$，而是学习一个低秩扰动：

$$
W' = W + \frac{\alpha}{r} \cdot BA, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times d}
$$

- $W$ 被冻结（无梯度、无优化器状态）
- 仅训练 $B$ 和 $A$：参数量为 $2 \times d \times r$ 而非 $d^2$
- 当秩 $r = 16$、$d = 4096$ 时：LoRA 每层仅增加 $2 \times 4096 \times 16 = 131\text{K}$ 个参数，而完整矩阵需 16.8M
- $\alpha/r$ 缩放因子控制更新的幅度

![图 1.10：LoRA 将权重更新 ΔW 分解为两个小矩阵 B × A。原始权重 W 保持冻结；只有 B 和 A 接收梯度。在推理时，乘积 BA 可零开销地合并入 W。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p74-09.png)

**为什么低秩有效**

Aghajanyan 等人 [93] 表明，微调发生在一个非常低维的子空间中——微调任务的「本征维度(intrinsic dimensionality)」远小于模型的参数量。一个 175B 模型的微调任务，其本征维度可能不到 1 万。LoRA 直接利用了这一点：秩 $r$ 将每个权重矩阵的更新约束在一个 $r$ 维子空间内。

**为什么 α/r 缩放很重要**

若不加缩放，将秩 $r$ 翻倍大致会使 $\Delta W = BA$ 的幅度也翻倍（B 中有更多列对求和有贡献）。这意味着改变秩也会改变模型被扰动的程度——你每次调整 $r$ 都需要重新调学习率。

$\alpha/r$ 因子对更新幅度进行归一化，使其无论秩为何都大致保持恒定：

$$
W' = W + \frac{\alpha}{r} \cdot BA
$$

- 固定 $\alpha$，扫描 $r$：有效更新幅度始终约为 $\alpha$，与秩无关。可以在 $r \in \{8, 16, 32, 64\}$ 中尝试而无需重新调 LR。
- 常见做法：设 $\alpha = r$（即 $\alpha/r = 1$）或 $\alpha = 2r$（即 $\alpha/r = 2$）。这是一种方便的默认设置，使缩放因子为一个小整数。
- 为何不直接调 LR？可以，但 $\alpha/r$ 提供了一个与秩无关的旋钮。团队可以在不同秩的实验之间共享 LR 配方。
- rsLoRA 洞察 [94]：在高秩（$r \ge 64$）下，经验证据表明 $\alpha/\sqrt{r}$ 比 $\alpha/r$ 更稳定，因为 $BA$ 的方差按 $\sqrt{r}$（而非 $r$）缩放。

### 1.9.2 LoRA 超参数

正确选择 LoRA 超参数至关重要——秩或 alpha 设置不当会导致欠拟合（约束过强）或浪费内存（表达能力过强）。

表 1.12：LoRA 超参数指南。

| 超参数 | 典型取值 | 指引 |
|---|---|---|
| r（秩） | 8, 16, 32, 64 | 越高容量越大但内存越多。从 16 起步。 |
| lora_alpha | 16, 32（常等于 r 或 2r） | 通过 α/r 缩放控制更新幅度。 |
| target_modules | q_proj, k_proj, v_proj, o_proj | 所有注意力投影。加入 gate_proj、up_proj、down_proj 以全覆盖。 |
| lora_dropout | 0.0–0.1 | 正则化。小数据集通常取 0.05。 |
| bias | "none" | 训练偏置仅增加极少量参数，但很少有益。 |
| Learning rate（学习率） | 1e-4 至 3e-4 | 高于全量微调（只有 adapter 在更新）。 |

**秩选择经验法则**

- r=8：简单任务（单领域对话、分类）。内存效率极高。
- r=16：通用微调。良好的默认值。
- r=32–64：复杂任务（数学、代码、多轮推理）。接近全量微调质量。
- r=128+：收益递减；考虑全量微调或更高秩的 QLoRA。
- 关键指标：若训练损失远高于全量微调损失并趋于平缓，则提高秩。

### 1.9.3 LoRA 变体

表 1.13：LoRA 变体及其创新点。

| 方法 | 关键创新 | 适用场景 |
|---|---|---|
| QLoRA [95] | 4-bit 量化的基座 + BF16 中的 LoRA。NF4 数据类型 + 双重量化。 | 在单张 48GB GPU 上微调 70B。 |
| DoRA [96] | 将 W 分解为幅度与方向；LoRA 仅更新方向。 | 推理任务上更好的泛化。 |
| LoRA+ [97] | A/B 使用不同 LR（$\eta_B = \lambda \eta_A$，$\lambda \approx 16$）。 | 免费 2% 提升；无额外成本。 |
| AdaLoRA [98] | 跨层动态分配秩预算（基于 SVD 的重要性）。 | 算力预算极紧。 |
| rsLoRA [94] | 按 $\alpha/\sqrt{r}$ 而非 $\alpha/r$ 缩放。 | 在高秩下稳定。使用 $r \ge 64$ 时。 |
| VeRA [99] | 共享的冻结随机 A、B；仅训练对角缩放。 | 极致参数效率。 |
| LoRA-FA | 初始化后冻结 A；仅训练 B。减半 LoRA 内存。 | 内存受限场景。 |

**关键扩展详解**

**DoRA——权重分解的低秩自适应(Weight-Decomposed Low-Rank Adaptation)。** DoRA [96] 观察到，全量微调倾向于改变权重向量的方向多于其幅度。标准 LoRA 将二者混为一谈。DoRA 将每个权重列分解为幅度 $m = \|W\|_{\text{col}}$ 与方向 $\hat{V} = W / \|W\|_{\text{col}}$，然后仅对方向施加 LoRA：

$$
W' = m \odot \hat{V}', \quad \hat{V}' = \frac{W + BA}{\|W + BA\|_{\text{col}}}
$$

幅度 $m$ 是一个独立的可学习向量（每列一个标量）。该方法在推理与指令遵循基准上始终比 LoRA 高 1–3%，且无额外推理成本（部署时合并）。

**LoRA+——非对称学习率(Asymmetric Learning Rates)。** Hayou 等人 [97] 表明，LoRA 中的矩阵 $A$ 和 $B$ 具有不同的最优学习率。由于 $B$ 初始化为零，它的起始状态与 $A$（从 $\mathcal{N}(0, \sigma^2)$ 初始化）截然不同。设 $\eta_B \approx 16 \times \eta_A$ 可将收敛速度和最终质量提升约 2%——这是一项仅靠一行配置改动即可获得的免费收益：

```python
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if "lora_B" in n],
     "lr": 2e-4 * 16},          # B 矩阵：更高 LR
    {"params": [p for n, p in model.named_parameters() if "lora_A" in n],
     "lr": 2e-4},               # A 矩阵：基础 LR
]
```

**VeRA——基于向量的随机矩阵自适应(Vector-based Random Matrix Adaptation)。** VeRA [99] 将参数效率推向极致：它不学习 $A$ 和 $B$，而是将它们作为所有层共享的冻结随机矩阵，仅训练两个对角缩放向量 $d_b \in \mathbb{R}^r$ 和 $d_a \in \mathbb{R}^d$：

$$
\Delta W = B \cdot \text{diag}(d_b) \cdot A \cdot \text{diag}(d_a)
$$

这将可训练参数比 LoRA 减少约 10 倍（每层仅 $r + d$ 个参数），同时达到 LoRA 质量的 90–95%。最适合需要数百个任务专用 adapter 且存储极少的场景。

**QLoRA 内存节省**

70B 模型全量微调：140 GB（权重）+ 280 GB（优化器）+ 140 GB（梯度）= 560 GB（7 张 A100-80GB）。

70B QLoRA（r=16，所有线性层）：

- 基座模型以 NF4 存储：$70\text{B} \times 0.5 = 35$ GB
- BF16 中的 LoRA adapter：约 160 MB
- 优化器状态（仅 adapter）：约 320 MB
- 激活（梯度检查点）：约 8 GB
- 总计：约 44 GB——可装入单张 48GB GPU！

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4 - 对权重最优
    bnb_4bit_compute_dtype=torch.bfloat16,  # 以 BF16 计算
    bnb_4bit_use_double_quant=True,         # 对量化常数再做量化
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,                           # alpha/r = 2x 缩放
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

```python
model = prepare_model_for_kbit_training(model)  # 为 QLoRA 准备
model = get_peft_model(model, lora_config)      # 添加 LoRA adapter
model.print_trainable_parameters()
```

### 1.9.4 其他 PEFT 方法

LoRA 主导了现代实践，但它并非唯一的参数高效方法。为完整起见，列出主要替代方案：

表 1.14：PEFT 方法族。LoRA 是 LLM 微调的事实标准；其余列入以提供历史背景与特定小众场景。

| 方法 | 机制 | 优点 / 缺点 | 状态 |
|---|---|---|---|
| LoRA [92]（及变体） | 在现有权重上添加低秩矩阵 | 推理时可合并（零开销）；支持完善；适用于所有架构 | 标准 |
| Adapters [100] | 在层间插入小型瓶颈 MLP | 模块化、可堆叠；增加推理延迟（额外的串行层） | 很少使用 |
| Prefix Tuning [101] | 在每层的键/值前预置可学习的「虚拟 token」 | 不修改权重；对生成任务有效；消耗上下文长度 | 小众 |
| Prompt Tuning [102] | 在输入前预置可学习的软提示嵌入 | 参数极少（<0.01%）；复杂任务上弱于 LoRA | 小众 |
| IA3 [103] | 学习对键、值和 FFN 激活进行缩放的向量 | 参数比 LoRA 更少；可合并；容量有限 | 已弃用 |
| BitFit [104] | 仅训练偏置项 | 近乎零参数；简单任务上出奇有效；表达能力有限 | 历史 |

**LoRA 为何胜出**

LoRA 成为标准，因为它独特地结合了：(1) 零推理开销——adapter 合并入基座权重，不像 Adapters 或 Prefix Tuning 那样增加延迟或消耗上下文；(2) 可组合性——可在服务时切换多个 LoRA adapter，实现多租户部署；(3) 生态支持——HuggingFace PEFT、TRL、vLLM 以及所有主流框架都有一流的 LoRA 支持；(4) 大规模验证——被 Meta、Google 以及 HuggingFace 上大多数开源微调用于生产。除非你有 LoRA 无法满足的特定约束，否则它应是你的默认选择。

## 1.10 混合专家（MoE）

混合专家(Mixture of Experts)模型 [105, 106] 通过对每个 token 仅激活一部分参数，在不按比例增加算力成本的情况下扩展模型容量。

### 1.10.1 架构

**MoE 层**

在 MoE Transformer 中，每个块中的 FFN 层被替换为 $N$ 个并行的「专家」FFN，外加一个选择使用哪些专家的路由器：

$$
\text{MoE}(x) = \sum_{i=1}^{N} g_i(x) \cdot E_i(x), \quad g(x) = \text{TopK}(\text{softmax}(W_r x))
$$

- $E_i$ 是专家网络（标准 FFN 层）
- $g_i(x)$ 是来自路由器的门控权重（只有 top-K 非零）
- 通常每个 token 在 $N = 8\text{–}64$ 个专家中激活 $K = 2$ 个
- 总参数随 $N$ 缩放；激活参数按 FFN 大小的 $K/N$ 缩放

![图 1.11：具有 8 个专家和 Top-2 路由的 MoE 层。每个 token 仅计算门控值最高的两个专家；其余专家完全跳过。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p79-10.png)

### 1.10.2 负载均衡

**负载均衡问题**

若无约束，路由器可能将大多数 token 都送到相同的 1–2 个专家（「专家坍塌 / expert collapse」）。这会浪费容量并造成跨 GPU 的算力不均衡（每个专家通常驻留在不同的 GPU 上）。

解决方案：添加一个辅助负载均衡损失：

$$
\mathcal{L}_{\text{bal}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot p_i
$$

其中 $f_i$ = 路由到专家 $i$ 的 token 比例，$p_i$ = 专家 $i$ 的平均路由器概率。这鼓励均匀的专家利用率。

### 1.10.3 噪声 Top-K 门控：使离散路由可训练

MoE 的核心挑战在于 top-k 选择不可微分——你无法对一个硬性的「选取前 2 个」操作进行反向传播。该领域发展出两个关键技巧来解决此问题：

**路由可微性问题**

路由器为每个专家计算 logit $h(x) = W_r \cdot x$，然后选取 top-k。但是：

- 被选中的专家通过其门控权重（在选中者之上的 softmax）获得梯度
- 选择决策本身（选哪 k 个）梯度为零
- 若无技巧，路由器会陷入困境：某个专家从未被选中 → 永远得不到梯度信号 → 永远不被选中

**方法 1：噪声 Top-K 门控 [105]。** 在 top-k 选择之前向路由器 logit 添加可学习的高斯噪声：

$$
h(x) = W_g \cdot x \quad \text{（干净 logit）}
$$

$$
H(x) = h(x) + \epsilon \cdot \text{Softplus}(W_{\text{noise}} \cdot x), \quad \epsilon \sim \mathcal{N}(0, 1) \quad \text{（带噪 logit）}
$$

$$
\text{TopK}(v, k)_i =
\begin{cases}
v_i & \text{若 } v_i \text{ 属于前 } k \\
-\infty & \text{否则}
\end{cases}
\quad (1.11)
$$

$$
g(x) = \text{softmax}\big(\text{TopK}(H(x), k)\big) \quad \text{（稀疏门控）}
$$

- $W_{\text{noise}}$ 是一个学到的噪声幅度——模型学习每个专家需要多少探索
- 训练期间，噪声偶尔将「冷门」专家推入 top-k，给予其梯度信号
- 推理时移除噪声：使用干净 logit $h(x)$ 进行确定性路由
- Softplus 确保噪声尺度始终为正

**方法 2：Gumbel-Softmax 技巧（用于可微分离散采样）。** 来自变分推断文献的一种替代方案 [107]。Gumbel-Max 技巧提供了从类别分布的精确采样：

$$
z = \arg\max_i \big[\log \pi_i + G_i\big], \quad G_i \sim \text{Gumbel}(0, 1)
\quad (1.12)
$$

其中 Gumbel 噪声生成为 $G_i = -\log(-\log(U_i))$，$U_i \sim \text{Uniform}(0, 1)$。

对于 top-k 路由：对 $(\log \pi_i + G_i)$ 取 top-k 即可从 $\pi$ 定义的类别分布中无放回地得到 $k$ 个样本。

由于 $\arg\max$ 不可微，Gumbel-Softmax 松弛用一个受温度控制的 softmax 来替代它：

$$
\hat{g}_i = \frac{\exp\big((\log \pi_i + G_i)/\tau\big)}{\sum_j \exp\big((\log \pi_j + G_j)/\tau\big)}
\quad (1.13)
$$

- $\tau \to 0$：趋近硬性 one-hot（精确但不可微）
- $\tau \to \infty$：趋近均匀（可微但无信息量）
- 实践中，在训练期间将 $\tau$ 从 1.0 退火到 0.1–0.5
- 直通估计器(straight-through estimator)：前向传播使用硬 top-k，反向传播使用 Gumbel-Softmax 梯度——两全其美

**实践中使用哪种方法？**

- Sparsely-Gated MoE [105]、Mixtral [106]、DeepSeek-V2 [108]：使用带高斯噪声的噪声 Top-K。简单、有效、大规模验证。
- Switch Transformer [109]：简化为无噪声的 Top-1（仅依赖负载均衡损失）。
- 研究 / 较小规模的 MoE：部分使用 Gumbel-Softmax 实现完全可微路由，尤其是当学习路由本身是研究目标时。
- 关键洞察：两种方法都通过噪声注入解决同一问题（使离散选择可训练）。高斯噪声更简单；Gumbel 噪声对类别采样有更强的理论保证。

### 1.10.4 知名 MoE 模型

| 模型 | 总参数 | 激活参数 | 专家 | 创新 |
|---|---|---|---|---|
| Switch Transformer [109] | 1.6T | 100B | 128, Top-1 | 首个大规模 MoE；简化路由 |
| Mixtral 8x7B [106] | 47B | 13B | 8, Top-2 | 开放权重；匹敌 Llama-2 70B 质量 |
| DeepSeek-V2 [108] | 236B | 21B | 160, Top-6 | DeepSeekMoE，共享 + 路由专家 |
| Qwen-MOe [32] | 14.3B | 2.7B | 60, Top-4 | 细粒度专家以提升效率 |
| DBRX [110] | 132B | 36B | 16, Top-4 | 细粒度，每块 4 个专家 |

## 1.11 LLM 训练中的多样性

多样性——体现在训练数据、模型输出和优化轨迹中——对于防止模式坍塌(mode collapse)并确保稳健、通用的 LLM 至关重要。本节涵盖适用于所有 LLM 训练阶段的关键多样性机制。

### 1.11.1 采样多样性

**多样化生成的采样策略**

- **温度 $\tau$**：$P(x_i) \propto \exp(\text{logit}_i / \tau)$。$\tau$ 越高 = 分布越均匀 = 多样性越高。RLHF 生成中典型取值：$\tau = 0.7\text{–}1.0$。
- **Top-k**：仅从 k 个最高概率 token 中采样。防止退化的低概率 token。
- **Top-p（核/nucleus）**：从累积概率 $\ge p$ 的最小 token 集合中采样。自适应：模型不确定时更多样。
- **Min-p**：仅保留 $P \ge p_{\min} \times P_{\max}$ 的 token。比 top-k 更有原则。
- **频率/存在惩罚(frequency/presence penalty)**：惩罚已出现在响应中的 token。鼓励词汇多样性。

### 1.11.2 训练数据多样性

- **提示多样性**：覆盖不同领域、难度级别和格式。金发姑娘原则(Goldilocks principle)：提示的成功率应在 20–80%。
- **去重(Deduplication)**：移除近似重复的训练样本（MinHash、n-gram 重叠）。重复会导致对特定模式过拟合。
- **数据混合(Data mixing)**：使用温度加权采样或课程(curriculum)策略在任务/领域间取得平衡。

### 1.11.3 促进多样性的方法

| 方法 | 如何促进多样性 |
|---|---|
| 温度缩放 | 更高的 $\tau$ 使分布更平缓；更多 token 变得合理。 |
| Top-p / Min-p | 自适应阈值在模型不确定时允许更宽的采样。 |
| 频率惩罚 | 惩罚重复 token，迫使响应内词汇变化。 |
| 数据去重 | 从训练数据中移除近似重复，防止对特定模式过拟合。 |
| 多领域混合 | 跨领域的温度加权采样确保广泛覆盖。 |
| 语言化采样(Verbalized sampling) | 提示模型显式地用语言表达响应上的概率分布 [111]。参见 §7.5。 |

## 1.12 文本生成：解码方法

一个训练好的语言模型在每一步输出词汇表上的概率分布：$P(x_t \mid x_{<t})$。解码策略决定了我们如何从该分布中选取下一个 token。这一选择深刻影响输出质量、多样性和连贯性。

### 1.12.1 贪心解码(Greedy Decoding)

最简单的策略：始终选取概率最高的 token。

$$
x_t = \arg\max_{v \in V} P(v \mid x_{<t})
$$

直觉：就像在句子中总是取最显而易见的下一个词。「法国的首都是……」 → 「巴黎」（概率 0.92）。

优点：确定性、快速、无超参数。

缺点：产生重复、平庸的文本。会错过那些早期低概率 token 通向全局更优输出的高质量序列。无多样性。

### 1.12.2 束搜索(Beam Search)

并行维护 $B$（束宽，beam width）个部分假设，每个用 top-k token 扩展，并保留得分最高的 $B$ 条完整序列：

$$
\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i})
$$

引入长度归一化以避免偏向短序列：

$$
\text{score}_{\text{norm}}(y) = \frac{1}{|y|^\alpha} \sum_{i=1}^{|y|} \log P(y_i \mid y_{<i}), \quad \alpha \in [0.6, 1.0]
$$

直觉：就像在迷宫中同时探索多条路径，在每个岔路口只保留 $B$ 条最有希望的路径。

优点：比贪心找到更高似然的序列；适合存在单一「正确」输出的翻译和摘要。

缺点：对开放式生成仍倾向于平庸、重复的文本；算力为 $B$ 倍；所有束常常收敛到相似输出。

![图 1.12：B=2 的束搜索。每一步仅得分最高的 2 条部分序列存活（蓝色）。得分较低的候选被剪除（灰色）。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p83-11.png)

### 1.12.3 多样束搜索(Diverse Beam Search)

标准束搜索会产生近似重复的束。多样束搜索 [112] 将束分为 $G$ 组，并在组间添加不相似度惩罚：

$$
\text{score}_g(y_t) = \log P(y_t \mid y_{<t}) - \lambda \sum_{g' < g} \Delta(y_t, Y_{g'})
$$

其中 $\Delta$ 度量与较早组已选 token 的重叠（如汉明多样性 / Hamming diversity），$\lambda$ 控制多样性强度。

直觉：就像强制一个头脑风暴小组产生不同想法——每个子组若重复先前子组所说的内容会受到惩罚。

优点：产生真正不同的候选序列；适用于重排序流水线。

缺点：多样性惩罚可能损害单束质量；更多超参数（$G$、$\lambda$）。

### 1.12.4 Top-k 采样

仅从 k 个最可能的 token 中采样，重新分配概率质量：

$$
P'(v \mid x_{<t}) =
\begin{cases}
\dfrac{P(v \mid x_{<t})}{\sum_{v' \in \text{Top-k}} P(v' \mid x_{<t})} & \text{若 } v \in \text{Top-k} \\[6pt]
0 & \text{否则}
\end{cases}
$$

直觉：在「猫坐在……」之后，只考虑 top-k 个合理的续词（「垫子 / mat」、「地板 / floor」、「沙发 / couch」……），忽略极不可能的（「量子 / quantum」、「群岛 / archipelago」）。

优点：去除尾部噪声；实现简单。

缺点：固定的 k 对尖峰分布过于受限（浪费概率质量），对平坦分布又过于宽松（放入垃圾 token）。

### 1.12.5 Top-p（核/Nucleus）采样

从累积概率超过 $p$ 的最小 token 集合中采样：

$$
\text{Top-p} = \min\left(S \subseteq V : \sum_{v \in S} P(v \mid x_{<t}) \ge p\right)
$$

其中 token 按概率降序排列并依次加入，直到达到阈值 $p$。

直觉：自适应调整候选池大小。若模型很自信（「巴黎」达 95%），核非常小。若不确定（「这部电影……」），核扩展以包含多个合理形容词。

优点：自适应分布形状；广泛使用的默认值（$p = 0.9\text{–}0.95$）。

缺点：仍会包含核尾部的一些低质量 token；阈值是单一全局超参数。

![图 1.13：Top-p（核）采样：token 按概率排序并被纳入，直到累积质量达到 p = 0.9。核（深蓝）根据分布形状自适应其大小——此处 5 个 token 即足够。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p84-12.png)

**Top-k 与 Top-p 对比**

考虑预测下一个词：

- 「2 + 2 =」之后：分布是尖峰的——top-1 token（「4」）占 99% 质量。Top-k=50 会浪费地考虑 49 个错误答案。Top-p=0.9 正确地只选「4」。
- 「我喜欢吃」之后：分布是平坦的——许多食物都合理。Top-k=5 过于受限。Top-p=0.9 可能包含 50+ 个 token，匹配真实的不确定性。

Top-p 会自适应；top-k 不会。实践中常将二者结合：从 top-p 与 top-k 的交集中采样。

### 1.12.6 Min-p 采样

一种较新的替代方案，设置相对概率下限 [113]：

$$
\text{Min-p} = \left\{v \in V : P(v \mid x_{<t}) \ge p_{\min} \cdot \max_{v'} P(v' \mid x_{<t})\right\}
$$

仅保留概率至少为 top token 概率 $p_{\min}$ 倍的 token。

直觉：「只考虑那些至少有最佳 token 10% 可能性的 token」。若 top token 概率为 0.8，则仅高于 0.08 的 token 存活。若 top token 概率为 0.05（非常不确定），则高于 0.005 的 token 存活——自然地扩展候选池。

优点：随模型置信度自然缩放；在尖峰分布上比 top-p 退化的样本更少；单一且直观的参数。

缺点：较新、实战检验较少；尚未在所有推理框架中成为标准。

### 1.12.7 温度缩放(Temperature Scaling)

在应用任何采样策略之前，logit 被温度 $T$ 除：

$$
P_T(v \mid x_{<t}) = \frac{\exp(z_v / T)}{\sum_{v'} \exp(z_{v'} / T)}
$$

- $T < 1$：锐化分布 → 更确定、更聚焦的输出。
- $T = 1$：未修改的模型分布。
- $T > 1$：平缓分布 → 更随机、更有创造力的输出。
- $T \to 0$：退化为贪心解码。$T \to \infty$：退化为均匀采样。

常用设置：事实类任务 $T = 0.7$，创意写作 $T = 1.0\text{–}1.2$，代码/数学 $T = 0.0$（贪心）。

### 1.12.8 对比解码(Contrastive Decoding)

对比解码 [114] 利用强模型（专家 / expert）与弱模型（业余 / amateur）之间的差异，放大专家独有的知识：

$$
x_t = \arg\max_{v \in V(x_{<t})} \big[\log P_{\text{expert}}(v \mid x_{<t}) - \log P_{\text{amateur}}(v \mid x_{<t})\big]
$$

其中 $V(x_{<t}) = \{v : P_{\text{expert}}(v \mid x_{<t}) \ge \alpha \cdot \max_{v'} P_{\text{expert}}(v' \mid x_{<t})\}$ 是一个自适应合理性约束。

直觉：业余模型捕获通用的、显而易见的模式（常用词、重复）。减去其对数概率可移除这种「通用信号」，留下专家独特的知识与推理。就像从录音中去除背景噪声以听清信号。

优点：减少重复和平庸措辞；无需额外训练即可改善事实性与连贯性；适用于任意模型对。

缺点：需要运行两个模型（2 倍算力）；对业余模型的选择敏感；合理性阈值 $\alpha$ 需要调参。

### 1.12.9 重复惩罚(Repetition Penalties)

与采样策略正交，重复惩罚抑制模型重复 token。给定 token $v$ 的原始 logit $z_v$（即 LM 头在 softmax 之前输出的未归一化分数），惩罚后的 logit 为：

$$
z'_v =
\begin{cases}
z_v / \theta & \text{若 } v \in \text{已生成 token 且 } z_v > 0 \\
z_v \cdot \theta & \text{若 } v \in \text{已生成 token 且 } z_v < 0
\end{cases}
$$

其中 $\theta > 1$ 是惩罚因子（通常 1.1–1.3）。在两种情况下，效果都是将 logit 推向零——降低先前已生成 token 的概率。频率与存在惩罚是 OpenAI API 使用的更简单的加性变体：

$$
z'_v = z_v - \alpha \cdot \text{count}(v) - \beta \cdot \mathbb{1}[v \in \text{已生成}]
$$

其中 $\alpha$ 是频率惩罚（与 $v$ 出现次数成正比），$\beta$ 是存在惩罚（对任何先前出现施加的固定惩罚）。

### 1.12.10 实践对比

表 1.15：LLM 文本生成解码方法对比。

| 方法 | 确定性 | 多样性 | 质量 | 最适用于 |
|---|---|---|---|---|
| 贪心(Greedy) | 是 | 无 | 中 | 代码、事实问答 |
| 束搜索（B=4–8） | 是 | 低 | 高（窄） | 翻译、摘要 |
| 多样束搜索 | 是 | 中 | 高 | 用于重排序的候选生成 |
| Top-k（k=50） | 否 | 中 | 中 | 通用生成 |
| Top-p（p=0.9） | 否 | 自适应 | 高 | 开放式任务的默认 |
| Min-p（pmin=0.1） | 否 | 自适应 | 高 | top-p 的稳健替代 |
| 对比(Contrastive) | 是 | 低 | 很高 | 事实性、连贯的长文 |

**实践中的解码：「Once upon a time」**

给定提示「Once upon a time,」：

- 贪心：「there was a young girl who lived in a small village...」（平庸的童话）
- Top-p=0.9, T=1.0：「the rivers ran backwards and the fish learned to fly...」（有创意、出人意料）
- Top-p=0.9, T=0.3：「there was a kingdom ruled by a wise and just king...」（连贯、常规）
- 对比：「in the amber-lit corridors of a collapsing star, two minds argued about the nature of time...」（独特、避免俗套）

同一模型、同一提示——解码策略决定了输出的特质。

### 1.12.11 约束解码（结构化生成 / Constrained Decoding）

上述所有方法在每一步都从完整词汇表中采样。约束解码限制允许的 token 集合，使得输出保证符合某种形式文法——通常是 JSON schema、正则表达式(regex)或上下文无关文法(CFG)。

**核心机制。** 在每个解码步 $t$，根据当前解析器状态计算一个 token 掩码 $M_t \subseteq V$。只有 $M_t$ 中的 token 保留其原始 logit；所有其余的在 softmax 之前被设为 $-\infty$：

$$
P'(v \mid x_{<t}) =
\begin{cases}
P(v \mid x_{<t}) / Z & \text{若 } v \in M_t \\
0 & \text{否则}
\end{cases}
$$

其中 $Z = \sum_{v \in M_t} P(v \mid x_{<t})$ 进行重新归一化。由于掩码每步都变（它取决于到目前为止已生成的内容），约束被增量式地执行——模型在任何时刻都无法产生非法前缀。

**从 schema 到掩码。** 编译流水线为：

$$
\text{JSON Schema} \xrightarrow{\text{compile}} \text{Regex} \xrightarrow{\text{compile}} \text{FSM (DFA)} \xrightarrow{\text{index}} \text{每状态 Token 掩码}
$$

FSM 状态对应正则中的位置。对每个状态，预先计算出所有能让字符串保持在文法语言内的词汇表 token，并索引化（每个 schema 仅一次性成本）。

## 1.13 提示工程（Prompt Engineering）

提示工程是设计 LLM 输入、从而可靠地激发目标行为的一门学科——它无需改变模型权重。微调修改的是模型本身，而提示工程则通过精心的框架设计、示例与结构来利用模型已有的能力。它是改善 LLM 输出最快、最廉价、也最容易上手的手段，即便在使用了微调模型时依然不可或缺。

### 1.13.1 上下文学习（ICL）

上下文学习（In-Context Learning, ICL）[21] 是大语言模型的一项惊人能力：模型能够在推理时仅凭提示中提供的示例就学会完成任务——无需任何梯度更新。模型会从输入–输出对的模式中隐式推断出任务，并泛化到新的输入。

**为什么上下文学习有效**

- **隐式贝叶斯推断（Implicit Bayesian inference）**：模型在预训练中见过数百万个任务，提示中的示例会在模型已学到的分布中定位到相关任务 [117]。
- **归纳头（Induction heads）**：特定的注意力头学会了复制模式（"A 之于 B 犹如 C 之于 ___"），从而实现上下文内的泛化 [67]。
- **任务向量（Task vectors）**：ICL 会在残差流（residual stream）中创建隐式的任务表示，将生成引导至示例所展示的格式与内容 [118]。

**规模化行为。** ICL 主要在约 10 亿参数以上的模型中涌现，并随模型规模呈对数线性提升 [21]。较小的模型能记住示例，却难以在同一上下文窗口内泛化到新输入。

### 1.13.2 零样本提示（Zero-Shot Prompting）

零样本提示不提供任何示例——只给出任务描述或指令。模型必须完全依赖其预训练知识与指令微调（instruction-tuning）来产生正确的格式与内容。

**零样本分类（Zero-Shot Classification）**

```
Classify the following movie review as POSITIVE or NEGATIVE.
Review: "The cinematography was breathtaking but the plot felt rushed and predictable."
Sentiment:
```

**零样本何时表现良好：**

- 模型在预训练/SFT 中大量见过的任务（翻译、摘要、情感分析）
- 输出格式明确无歧义、指令表述清晰
- 指令微调模型（如 ChatGPT、Claude、Llama-3-Instruct）在零样本任务上显著优于基座模型 [9]

**零样本何时失效：**

新格式、领域特定的标注方案，或是模型无法仅凭指令就推断出你确切需求的歧义任务。

### 1.13.3 少样本提示（Few-Shot Prompting）

少样本提示 [21] 在实际查询之前提供 $k$ 个输入–输出示例（"shots"）。这是上下文学习最常见的形式，也始终是最有效的提示策略之一。

**少样本命名实体识别（Few-Shot Named Entity Recognition）**

```
Extract named entities from the text. Format: [ENTITY](TYPE)
Text: "Apple released the iPhone 15 in Cupertino."
Entities: [Apple](ORG), [iPhone 15](PRODUCT), [Cupertino](LOC)
Text: "Elon Musk announced Tesla's new factory in Berlin."
Entities: [Elon Musk](PER), [Tesla](ORG), [Berlin](LOC)
Text: "OpenAI partnered with Microsoft to deploy GPT-4."
Entities:
```

少样本示例的关键设计原则：

1. **多样性（Diversity）**：覆盖预期输入的范围（不同长度、边界情况、各类别）。
2. **顺序（Ordering）**：把更难或更具代表性的示例放在最后（利用近因偏差）[119]。
3. **标签均衡（Label balance）**：若是分类任务，应包含所有类别的示例，以避免多数类偏差。
4. **格式一致性（Format consistency）**：每个示例都必须遵循完全相同的结构，模型会模仿这一模式。
5. **相关性（Relevance）**：使用与目标查询语义相似的示例效果最佳 [120]。

**用几个示例？** 性能通常在 0 到 4–8 个示例之间提升，随后趋于平台期。超过约 20 个示例时收益甚微，且有填满上下文窗口的风险。Min 等人 [121] 证明，示例的格式与标签空间比标签正确性更重要——即便是随机标签也有帮助（不过正确标签帮助更大）。

### 1.13.4 指令遵循型提示（Instruction-Following Prompts）

指令微调模型对清晰、结构化的指令响应最佳。关键洞察是：把提示当作一份规格说明（specification），而不是一个建议。

**有效指令提示的构成**

1. **角色/人设（Role/Persona）**：定义模型是谁（"你是一位资深数据科学家……"）
2. **任务（Task）**：清晰、无歧义地说明要做什么
3. **上下文（Context）**：模型所需的背景信息
4. **约束（Constraints）**：长度限制、语气、要避免什么、输出格式
5. **示例（可选）**：展示期望的输出格式
6. **输入（Input）**：待处理的实际数据

**带约束的指令提示（Instruction Prompt with Constraints）**

```
Role: You are a medical literature reviewer.
Task: Summarize the following research abstract for a general audience.
Constraints:
- Maximum 3 sentences
- No jargon (explain any technical terms)
- Include the key finding and its clinical implication
- Do NOT speculate beyond what the abstract states
Abstract: [...]
```

**系统提示与用户提示（System prompts vs. user prompts）。** 现代聊天 API 将系统提示（system prompt，持久指令、角色定义）与用户消息（user message，每轮输入）分开。在大多数模型中，系统提示会被以更高的注意力优先级处理，是放置角色定义、约束和输出格式规格的自然位置 [23]。

### 1.13.5 结构化输出提示（Structured Output Prompts，JSON/XML）

对于程序化使用而言，最关键的提示技术是强制结构化输出——尤其是 JSON。

**JSON 输出提示（JSON Output Prompt）**

```
Extract the following information from the customer email.
Respond ONLY with valid JSON, no other text.
Schema:
{
  "intent": "refund|complaint|question|praise",
  "urgency": "low|medium|high",
  "product_mentioned": "string or null",
  "summary": "one sentence summary"
}
Email: [...]
```

可靠结构化输出的技术：

- **模式优先（Schema-first）**：在输入之前展示确切的 JSON 模式，模型会将其当作模板。
- **约束解码（Constrained decoding）**：使用基于语法的采样（如 Outlines [115]、Guidance）在 token 层面保证 JSON 的语法有效性。
- **XML 标签（XML tags）**：对于嵌套或多部分的输出，XML 标签（如 `<thinking>...</thinking>`）提供了模型能可靠遵循的明确分隔符。
- **Pydantic / TypeScript 类型**：提供类型定义有助于模型理解字段约束（OpenAI 的函数调用内部就使用 JSON Schema）。

**提示中的 JSON——常见陷阱（Common Pitfalls）**

- 模型可能添加 markdown 围栏（```json ... ```）——要明确指示输出原始 JSON。
- 嵌套对象与数组会增加幻觉风险——尽量拍平（flatten）模式。
- 枚举字段（固定取值）远比自由文本字段可靠。
- 务必通过程序校验输出；没有约束解码时，没有任何提示能保证 100% 合规。

**JSON 提示：组织输入（JSON Prompting: Structuring the Input）。** 一种不同但互补的技术是 JSON 提示——把提示本身格式化为 JSON 而非自然语言。这利用了模型在结构化数据（API、配置、代码）上大量的预训练，从而提升指令遵循度、降低歧义，并支持对多字段请求的确定性解析。

**带系统提示的 JSON 提示（JSON Prompting with System Prompt）**

```
=== SYSTEM ===
You are a senior code reviewer. Analyze code for bugs, security issues, and style violations. Always respond in the JSON schema provided.
=== USER (JSON prompt) ===
{
  "task": "code_review",
  "language": "python",
  "severity_filter": "high",
  "code": "def login(user, pw):\n  query = ...",
  "output_schema": {
    "issues": [{
      "line": "int",
      "severity": "critical|high|medium|low",
      "category": "security|bug|style|performance",
      "description": "string",
      "fix": "string"
    }],
    "overall_risk": "critical|high|medium|low"
  }
}
```

**为什么 JSON 提示有效：**

- **无歧义的字段边界**：不会混淆一条指令在哪里结束、另一条从哪里开始。
- **类型化约束**：像 `"severity_filter": "high"` 这样的字段比"只显示高严重性 issue"更清晰。
- **模式即契约（Schema-as-contract）**：在输入中包含 `output_schema` 镜像了模型在预训练中大量见过的 API 设计模式。
- **系统提示仍然不可或缺**：系统消息提供了角色、语气和行为约束，这些很难自然地放进一个 JSON 负载中。

### 1.13.6 思维链（Chain-of-Thought, CoT）提示

思维链（Chain-of-Thought, CoT）提示 [122] 要求模型在给出最终答案之前先生成中间推理步骤。这一简单技术能显著提升那些需要多步推理的任务的表现：算术、逻辑、常识推断以及代码生成。

**为什么 CoT 有效：**

- **序列化计算（Serializes computation）**：Transformer 的深度固定，但生成的长度可变。CoT 把并行（困难）的问题转化为顺序（容易）的步骤，相当于扩展了模型的计算预算。
- **降低复合误差（Reduces compounding errors）**：每一步都是一个更简单的子问题，单步错误率更低。
- **暴露中间状态（Exposes intermediate state）**：使推理可审计、可调试。

**思维链变体（Chain-of-Thought Variants）**

| 方法 | 描述 |
|---|---|
| Zero-shot CoT [123] | 在任何提示后追加"Let's think step by step" |
| Few-shot CoT [122] | 提供带有显式推理链的示例 |
| Self-Consistency [124] | 采样 N 条 CoT 路径；对最终答案做多数投票 |
| Tree of Thoughts [125] | 探索多条推理分支，支持回溯 |
| Plan-and-Solve [126] | 先规划步骤，再逐步执行 |
| ReAct [127] | 交替进行推理与行动（工具使用） |

**零样本思维链（Zero-Shot Chain-of-Thought）**

```
Q: A store has 45 apples. They sell 3/5 of them in the morning and 1/3 of the remaining in the afternoon. How many apples are left?
Let's think step by step.
A: Morning sales: 45 * 3/5 = 27 apples sold.
   Remaining after morning: 45 - 27 = 18.
   Afternoon sales: 18 * 1/3 = 6 apples sold.
   Remaining: 18 - 6 = 12 apples.
```

**自洽性（Self-Consistency）。** Wang 等人 [124] 表明，采样多条思维链推理路径并对最终答案做多数投票，显著优于单路径 CoT。其直觉是：正确的推理路径倾向于收敛到同一答案，而错误通常是个别的（idiosyncratic）。这种做法以计算量（生成 $N$ 个样本）换取准确性——在延迟不如正确性重要时很实用。

**CoT 何时有害（When CoT hurts）。** CoT 并非普遍有益。对于简单任务（单步分类、检索、格式化），CoT 会增加不必要的 token、抬高延迟，甚至可能因"过度思考"引入错误。应在你预期需要多步推理的任务中选择性地使用 CoT。

### 1.13.7 高级提示技术（Advanced Prompting Techniques）

**检索增强生成（Retrieval-Augmented Generation, RAG）。** RAG [128] 不再仅依赖模型的参数化记忆，而是检索相关文档并将其纳入提示：

```
Context (retrieved): [document chunks]
Question: [user query]
Answer based ONLY on the provided context.
```

这将模型的回答锚定（ground）到可核验的来源上，能大幅降低知识密集型任务中的幻觉。

**提示链与分解（Prompt Chaining and Decomposition）。** 复杂任务适合拆解为一个由更简单提示组成的流水线，其中一步的输出成为下一步的输入：

1. 从文档中抽取关键事实
2. 在抽取的事实上进行推理
3. 格式化最终答案

每一步都可以使用不同的提示模板、模型或温度设置。这比单个巨型提示更可控，也支持针对性的调试。

**Constitutional AI / 自我批评（Self-Critique）。** Bai 等人 [129] 引入了一类提示，要求模型对照一组原则来批评并修订自身的输出：

```
[Generate initial response]
Critique: Does this response violate any of the following principles? [list principles]
Revision: Rewrite the response addressing the critique.
```

**元提示与提示优化（Meta-Prompting and Prompt Optimization）。** 与其手工打磨提示，近期的一些工作将提示设计自动化：

- **APE [130]**：用一个 LLM 自动生成候选提示并打分。
- **DSPy [131]**：将声明式任务描述编译为优化过的提示流水线，并附带学到的少样本示例。
- **OPRO [132]**：把提示优化视为一个优化问题，以 LLM 充当优化器。

**注意力推理查询（Attentive Reasoning Queries, ARQ）。** ARQ [133] 针对标准提示的一个根本弱点：随着上下文长度增长，模型会越来越多地在提示中部"丢失"关键信息（即 lost-in-the-middle 效应）。ARQ 通过将复杂查询分解为多个聚焦的子查询来缓解这一问题，每个子查询都旨在把模型的注意力导向上下文的特定部分：

1. **查询分解（Query decomposition）**：把用户问题拆成原子的子问题，每个子问题只针对一个狭窄方面。
2. **注意力检索（Attentive retrieval）**：为每个子查询仅检索或高亮相关的上下文切片——迫使模型关注它。
3. **聚合（Aggregation）**：把各子答案合并为连贯的最终回答。

这对长文档问答、大型检索集上的多跳（multi-hop）推理，以及上下文窗口中包含大量工具输出的智能体任务尤为有效。ARQ 可被视为思维链的一种结构化形式——它不仅管理模型如何推理，还显式地管理模型"看向哪里"。

### 1.13.8 最佳实践：打造有效提示（Best Practices: Crafting Effective Prompts）

基于文献中的实证发现与实践经验，以下原则能可靠地提升提示质量：

**提示工程清单（The Prompt Engineering Checklist）**

1. **具体且无歧义**：把"总结一下"换成"用 2–3 个要点总结，每条不超过 20 个词，聚焦可操作的发现"。
2. **展示而非告知（Show, don't tell）**：一个好示例抵得上 100 字的指令。拿不准时，加一个少样本示例。
3. **显式定义输出格式**：指定 JSON 模式、要点、表格格式或确切分隔符，永远不要把格式留给模型自由发挥。
4. **为输入数据使用分隔符**：用清晰的分隔符（`"""`、`<input>...</input>`、`---`）包裹用户输入，把指令与数据分开。
5. **赋予角色**："你是一位 [领域专家]，会 [具体行为]"能激活相关知识并定下语气。
6. **明确不要做什么**：负面约束（"不要解释你的推理"、"永远不要输出超过 5 项"）往往比正面约束更有效。
7. **为推理任务添加思维链**：在数学、逻辑或多跳问题后追加"逐步思考"，或提供完整示例。
8. **恰当控制温度**：事实性/确定性任务用 $T \approx 0$；创意性/多样化输出用 $T \approx 0.7$–$1.0$。
9. **实证迭代**：把提示当作代码——做版本管理、A/B 测试，并在有代表性的评测集上衡量表现。
10. **利用近因偏差（Recency bias）**：把最关键的指令和示例放在提示末尾（最靠近生成点处）。

**提示工程的心态（The Prompt Engineering Mindset）**

把提示工程视为用自然语言编程。模型是一个强大但刻板的解释器——它会严格按你所说的去做，并按其训练分布中最可能的方式去理解你的意图。软件工程中的通用原则同样适用：

**表 1.16：常见的提示失败模式与解决方案。**

| 失败模式 | 症状 | 解决方案 |
|---|---|---|
| 指令遗忘（Instruction amnesia） | 模型忽略长提示中的约束 | 把约束移到末尾；重复关键规则；使用系统提示 |
| 格式漂移（Format drift） | 输出开头正确但在长生成中退化 | 使用约束解码；拆成更短的链式提示 |
| 谄媚（Sycophancy） | 模型附和提示中的错误前提 | 加入"若不正确，请挑战假设"；使用系统级指令 |
| 编造细节（Hallucinated details） | 模型编造所提供上下文中没有的事实 | 加入"若不知道就说不知道"；使用带来源归属的 RAG |
| 拒绝误触发（Refusal over-triggering） | 模型因安全训练而拒绝良性请求 | 改写以澄清合法意图；明确说明该请求为何合理 |

- **DRY（Don't Repeat Yourself）**：除非为了对抗长上下文中的注意力衰减。
- **关注点分离（Separation of concerns）**：为角色、约束、示例和输入设置不同的提示区块。
- **测试驱动开发（Test-driven development）**：在编写提示之前先定义期望输出。
- **版本控制（Version control）**：跟踪提示的迭代及其评测分数。
- **模块化（Modularity）**：构建可复用的提示模板；将可变部分参数化。

当系统化迭代后提示仍无法达到所需质量时，那就是该转向微调（SFT）或强化学习（RLHF/DPO）的信号。

## 1.14 模型压缩方法（Model Compression Methods）

模型压缩旨在缩减模型规模与推理成本的同时保持质量。主要方法有三：量化（quantization，降低精度）、剪枝（pruning，移除参数）与蒸馏（distillation，训练更小的模型来模仿更大的模型）。

### 1.14.1 量化（Quantization）

量化通过以更低精度的格式表示权重（以及可选的激活值）来降低模型规模与推理成本。其核心权衡是压缩率与质量退化之间的平衡。

**量化概览（Quantization Overview）**

量化将模型权重（以及可选的激活值）的数值精度从 FP32/BF16 降到更低比特的格式：

$$
x_q = \text{round}\left(\frac{x - z}{s}\right), \qquad x_{\text{dequant}} = s \cdot x_q + z
$$

其中 $s$ 是缩放因子（scale factor），$z$ 是零点（zero-point）。

**何时量化（When to Quantize）**

- **推理服务**：始终量化。W4A16（权重 4 比特、激活 BF16）是最佳平衡点——对 70B 以上的模型可节省约 2 倍内存，质量损失不到 1%。
- **训练**：在 H100 上用 FP8 可获得约 2 倍吞吐量且质量损失极小；较小模型仍以 BF16 为默认。
- **边缘部署**：GGUF Q4_K_M 适合在消费级硬件上做本地推理。

**表 1.17：LLM 的量化方法。**

| 方法 | 比特数 | 类型 | 核心思想 |
|---|---|---|---|
| GPTQ [134] | 4-bit | PTQ，仅权重 | 逐层量化，通过 optimal brain surgeon 最小化 $\lVert WX - \hat{W}X \rVert^2$。 |
| AWQ [135] | 4-bit | PTQ，仅权重 | 保护显著权重（激活值较大的那些）。约 1% 的权重承载 99% 的重要性。 |
| GGUF [136] | 2–8 bit | PTQ，仅权重 | 面向 CPU 优化的格式（llama.cpp）。采用逐块量化，支持多种类型。 |
| FP8 (E4M3) | 8-bit | 训练 + 推理 | H100 原生支持。相比 BF16 约 2 倍吞吐。 |
| SmoothQuant [137] | W8A8 | PTQ，权重 + 激活 | 在量化前把激活值的离群点（outliers）平滑进权重，从而支持 INT8 GEMM。 |
| QAT [138] | 4-bit | QAT | 在模拟量化下训练。质量最高但开销大。 |
| AQLM [139] | 2-bit | PTQ，加性编码 | 通过学习到的加性量化码本实现极致压缩。 |

- **RLHF**：将冻结的模型（参考模型、奖励模型）量化为 INT8/FP8；策略模型保持 BF16 以保证训练精度。

### 1.14.2 剪枝（Pruning）

**为什么要剪枝（Why Prune）？** 现代 LLM 含有数十亿参数，但实证研究一再表明，其中很大一部分权重对模型输出的贡献微乎其微。剪枝正是利用这种过参数化（over-parameterization）：通过移除冗余权重，我们可以降低内存占用（从而能部署到更小的 GPU 或边缘设备）、推理延迟（每次前向传播的乘加运算更少）和服务成本（单位美元的吞吐量更高）。与量化对所有权重一致地降低精度不同，剪枝选择性地剔除权重——在与量化结合时能带来乘法级别的节省（例如一个 50% 稀疏的 4 比特模型，内存占用比稠密 BF16 基线少 4 倍）。挑战在于如何在保持生成质量的前提下达到高稀疏度，这推动了无需重训练的有原则的一次性（one-shot）方法的发展。

**剪枝方法（Pruning Methods）**

- **非结构化剪枝（Unstructured pruning）**：将低于阈值的单个权重置零。可达高稀疏度（50–90%）。需要稀疏 GEMM 内核（A100/H100 上的 2:4 模式）。
- **结构化剪枝（Structured pruning）**：移除整个注意力头、层或 FFN 神经元。无需专用内核即可直接降低 FLOPS。
- **SparseGPT [140]**：使用近似逆 Hessian 的一次性剪枝。在 175B 模型上达到 50% 非结构化稀疏度，质量损失极小。
- **Wanda [141]**：按 $|w| \times \lVert x \rVert$（权重幅度乘以输入激活范数）剪枝。无需校准数据，与 SparseGPT 表现相当。

**NVIDIA 2:4 结构化稀疏（NVIDIA 2:4 Structured Sparsity）**

A100/H100 的 Tensor Core 原生支持 2:4 稀疏：每 4 个元素中至多有 2 个非零。在受支持的操作上可借助硬件加速获得恰好 2 倍的加速。约束是：必须在这种特定模式下达到恰好 50% 的稀疏度，灵活性不如任意稀疏度。

### 1.14.3 知识蒸馏（Knowledge Distillation）

知识蒸馏（Knowledge Distillation）[142] 把大型教师模型（teacher）已学到的行为迁移到更小、更廉价的学生模型（student）中。核心思想是：教师在 token 上的输出分布所携带的信号，远比仅有 ground-truth 的硬标签（hard labels）丰富得多——它揭示了类别间相似性、校准（calibration）和不确定性，学生可以加以利用。

**带温度的 softmax（Temperature-Scaled Softmax）。** 为了暴露教师 logits 中的"暗知识（dark knowledge）"，我们用温度 $T > 1$ 来软化分布：

$$
p^{(T)}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

在高温下，概率质量会分布到更多 token 上，使那些"差一点"的备选项变得可见。训练时对学生施加同样的温度；推理时学生使用 $T = 1$。

**通用蒸馏损失（General Distillation Loss）。**

$$
\mathcal{L}_{\text{distill}} = \alpha \, T^2 \cdot \mathrm{KL}\!\left(P^{(T)}_{\text{teacher}} \;\|\; P^{(T)}_{\text{student}}\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}\!\left(y, P^{(1)}_{\text{student}}\right)
$$

$T^2$ 系数用于补偿软化分布造成的梯度幅度下降。典型取值：$T \in [2, 20]$，$\alpha \in [0.5, 0.9]$（教师质量越高，KL 项权重越大）。

**表 1.18：LLM 的知识蒸馏范式。**

| 范式 | 机制 | 优点 | 缺点 |
|---|---|---|---|
| 离线 / 白盒（Offline / White-box） | 教师 logits 预先计算；学生在完整分布上训练 | 完整分布信号；一次性教师成本 | 数据陈旧；存储开销大 |
| 在线 / 协同训练（Online / Co-training） | 教师实时生成；学生看到新鲜的 logits | 能适应学生的弱点 | 2 倍计算量；基础设施复杂 |
| 黑盒（Black-box, API） | 仅有教师文本输出（无 logits） | 适用于专有模型 | 丢失暗知识；类似 SFT |
| 自蒸馏（Self-distillation） | 模型蒸馏到更小版本的自己 | 无需单独的教师 | 教师 = 学生同族；受天花板限制 |

**离线（白盒）蒸馏（Offline (White-Box) Distillation）。** 对每个训练 token，记录教师的完整 logit 向量（或为节省存储而记录 top-k logits）。学生最小化与这些存储分布之间的 KL 散度。在教师访问不受限时，这是数据效率最高的范式。

- **动机**：把教师推理与学生训练解耦——在高端硬件上跑一次教师，然后低成本训练多个学生。
- **优点**：确定且可复现；教师成本被摊销；获得完整分布信号。
- **缺点**：需要为每个 token 存储 $|V|$ 维向量（可用 top-k 剪枝缓解）；教师无法针对学生的失败做适配。

**在线（协同训练）蒸馏（Online (Co-Training) Distillation）。** 教师与学生联合运行：教师为学生当前训练批次生成 logits。

- **动机**：让教师聚焦于学生当前挣扎的输入（类似课程学习）。
- **优点**：新鲜度高；可用学生生成的输入做 on-policy 蒸馏。
- **缺点**：GPU 成本翻倍；同步复杂；更难扩展。

**黑盒（API）蒸馏（Black-Box (API) Distillation）。** 当只有文本输出可用时（例如从专有 API 蒸馏），学生通过在教师生成内容上做 SFT 来训练，可选地辅以思维链轨迹。

- **动机**：现实考量——大多数前沿模型不暴露 logits。
- **优点**：流水线简单；适用于任何 API 背后的模型。
- **缺点**：没有软标签信号；容易出现幻觉放大；实质上就是监督微调。

**自蒸馏（Self-Distillation）。** 模型从同一架构家族中更大的版本（如 Llama-3 70B → 8B）蒸馏，或从自身训练过程中的检查点蒸馏。

- **动机**：避免训练单独的教师；利用模型自身在不同规模上的能力。
- **优点**：架构兼容；无外部依赖。
- **缺点**：教师天花板等于模型自身天花板；无法引入真正的新知识。

**暗知识（Dark Knowledge）**

考虑一个在"The capital of France is"之后预测下一个词的语言模型。硬标签只说"Paris"是对的。但教师的软分布可能给"Lyon"分配 5%、给"Marseille"分配 2%、给"banana"分配接近 0%——这就告诉学生哪些错误是合理的，从而显著改善校准与泛化。

**LLM 蒸馏的实践考量（Practical Considerations for LLM Distillation）。**

- **序列级 vs. token 级**：token 级 KL 是标准做法；序列级蒸馏（在全序列上最小化 KL）能更好地捕捉长程连贯性，但更难优化。
- **逐层提示（Layer-wise hints）**：匹配中间表示（注意力图、隐藏状态）可提供额外的学习信号——在学生架构与教师不同时尤其有用。
- **数据选择**：蒸馏数据质量至关重要；精选多样化、困难样本比随机采样能产出更好的学生模型。
- **学生容量**：在低于约教师参数量 10% 时收益递减；在极端压缩下，可能需要改变架构（如 MoE → 稠密）。
- **与量化结合**：蒸馏 + 4 比特量化（如 QLoRA 蒸馏的模型）可在 20 倍压缩下达到接近教师的质量。

**压缩方法对比——70B 模型（Compression Method Comparison – 70B Model）**

| 方法 | 大小 | 速度 | 质量 | 用途 |
|---|---|---|---|---|
| BF16（基线） | 140 GB | 1× | 100% | 训练、参考 |
| FP8 (E4M3) | 70 GB | 2× | 99.5% | H100 推理 |
| INT8 (SmoothQuant) | 70 GB | 1.8× | 99% | A100 推理 |
| 4-bit (AWQ) | 35 GB | 2.5× | 97–98% | 大规模服务 |
| 2-bit (AQLM) | 17.5 GB | 3× | 90–93% | 边缘、实验性 |
| 50% 剪枝 (2:4) | 70 GB | 1.8× | 97% | 结构化加速 |
| 蒸馏 8B | 16 GB | 10× | 80–85% | 移动、边缘 |

## 1.15 推测解码方法（Speculative Decoding Methods）

推测解码（speculative decoding）[143] 通过同时预测多个 token、然后用目标模型在一次前向传播中加以验证，来加速自回归生成。它产出的输出分布与标准解码完全相同（无质量损失），同时可获得 2–3 倍加速。

### 1.15.1 核心原理（Core Principle）

**推测解码框架（Speculative Decoding Framework）**

1. 一个快速的草稿机制提出 $k$ 个候选 token：$\hat{x}_1, \dots, \hat{x}_k$
2. 大型目标模型对所有 $k$ 个 token 运行单次前向传播（批处理）
3. **验证**：从左到右接受满足 $P_{\text{target}}(\hat{x}_i) \geq r_i \cdot P_{\text{draft}}(\hat{x}_i)$ 的 token（其中 $r_i \sim U[0,1]$）
4. 在位置 $j$ 首次被拒绝时：从调整后的分布中重新采样 $x_j$，并丢弃 $\hat{x}_{j+1}, \dots, \hat{x}_k$

**关键性质**：这种接受/拒绝方案保证最终分布精确等于 $P_{\text{target}}$。

**加速**：若接受率为 $\alpha$，则每步期望 token 数为 $\frac{1 - \alpha^{k+1}}{1 - \alpha}$。在 $\alpha = 0.8$、$k = 5$ 时，期望每步 3.4 个 token，而标准解码仅为 1。

### 1.15.2 方法对比（Methods Comparison）

**表 1.19：现代推理引擎支持的推测解码方法。**

| 方法 | 草稿来源 | 加速 | 核心思想 |
|---|---|---|---|
| Standard [143] | 小模型（1–7B） | 2–3× | 用独立的草稿模型生成候选。简单但需加载 2 个模型。 |
| Medusa [144] | 并行 LM 头 | 2–3× | 在目标模型上增加 $k$ 个额外的预测头，每个预测位置 +1、+2、…、+k 的 token。 |
| Eagle [145] | 特征级 | 2.5–3.5× | 轻量解码器从目标模型的隐藏状态生成草稿 token。接受率高于 Medusa。 |
| Eagle-2 [145] | 上下文感知 | 3–4× | 基于置信度动态扩展的草稿树。达到业界领先的接受率。 |
| N-gram Lookup | N-gram 缓存 | 1.5–2× | 将提示的 n-gram 与此前生成的文本进行匹配。零成本；对重复性输出效果好。 |
| Lookahead [146] | 雅可比迭代 | 2–2.5× | 并行雅可比解码 + n-gram 验证。无需草稿模型，直接用目标模型。 |
| Multi-token [147] | 改造的架构 | 2–3× | 训练模型原生地每步预测多个 token（Meta 在 Llama 中的做法）。 |

### 1.15.3 Medusa：多头推测解码（Medusa: Multi-Head Speculative Decoding）

**Medusa 如何工作（How Medusa Works）**

Medusa 在 LLM 上额外增加 $k$ 个"预测头"（共享同一主干）：

- **Head 0（原始）**：预测位置 $t+1$ 的 token（标准下一 token）
- **Head 1**：预测位置 $t+2$ 的 token（跳过一格）
- **Head $i$**：预测位置 $t+i+1$ 的 token
- 所有头在单次前向传播中并行运行
- 一个树状验证机制同时校验多条候选序列

**训练**：仅微调 Medusa 的头（主干冻结）。成本：在代表性数据上约 1 个 epoch。

**优势**：无需单独的草稿模型；各个头都很小（每个仅一层线性层）。**内存开销**：<1%。

### 1.15.4 Eagle：特征级草稿（Eagle: Feature-Level Drafting）

**为什么 Eagle 优于 Medusa（Why Eagle Outperforms Medusa）**

Medusa 的各头在每个位置独立预测——它们无法依赖自己此前的预测（位置 $t+2$ 的 token 并不知道 $t+1$ 处预测了什么）。Eagle 用一个轻量的自回归解码器在目标模型的隐藏状态上工作，修复了这一问题：

1. 从目标模型最后一层抽取隐藏状态
2. 输入到一个小的（1 层）解码器，该解码器以先前的隐藏状态为条件自回归地生成草稿 token
3. 这捕获了 Medusa 所遗漏的 token 间依赖关系

**结果**：Eagle 的接受率达到 85–95%，而 Medusa 为 60–80%。

### 1.15.5 N-gram 推测解码（N-gram Speculative Decoding）

**N-gram 查找方法（N-gram Lookup Method）**

最简单的推测解码——无需额外模型或训练：

1. 维护一个来自提示与此前生成文本的 n-gram 缓存
2. 每一步，检查当前上下文最后 $n-1$ 个 token 是否匹配某个缓存的 n-gram
3. 若匹配：将该续接作为草稿 token 提出
4. 照常用目标模型验证

**最适用于**：代码生成（重复模式）、结构化输出（JSON/XML），以及含重复元素的提示。**成本**：几乎为零。

### 1.15.6 与 vLLM 集成（Integration with vLLM）

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=4,
    speculative_config={
        "model": "meta-llama/Llama-3-8B",
        "num_speculative_tokens": 5,
    },
)

llm = LLM(
    model="meta-llama/Llama-3-70B",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,   # 最多从提示中匹配 4-gram
    },
)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 2,
        "method": "eagle",
        "draft_tensor_parallel_size": 1,
    },
)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "ibm-ai-platform/llama3-70b-accelerator",
        "draft_tensor_parallel_size": 1,
    },
)
```

**何时不该用推测解码（When NOT to Use Speculative Decoding）**

- **大批次（High batch sizes）**：当 batch $\geq 64$ 时，生成本身已具计算效率。推测带来的额外开销（草稿生成 + 验证）得不偿失。
- **分布差异过大（Very different distributions）**：若草稿模型与目标差异太大，接受率跌破 50%，推测反而比标准解码更慢。
- **短输出（Short outputs）**：对于 <20 token 的输出，推测的启动成本超过收益。
- **经验法则**：推测对延迟敏感的单流式生成（聊天机器人、交互式代码补全）帮助最大。

## 1.16 幻觉检测（Hallucination Detection）

LLM 会生成读起来通顺但事实可能不正确的文本——这种现象称为幻觉（hallucination）[148]。本节介绍模型层面（model-level）的基础检测方法（不依赖外部检索或多智能体验证）。

### 1.16.1 幻觉的类型（Types of Hallucination）

**幻觉分类（Hallucination Taxonomy）**

- **内在幻觉（Intrinsic）**：与所提供的输入/上下文相矛盾（例如摘要说出了与原文相反的意思）。
- **外在幻觉（Extrinsic）**：生成了无法从输入中核验且事实上有误的断言。
- **忠实性（Faithfulness）**：输出偏离了指令或指定的约束。

### 1.16.2 检测方法（模型层面）（Detection Methods, Model-Level）

**表 1.20：在模型层面运行的基础幻觉检测方法。**

| 方法 | 机制 | 信号 |
|---|---|---|
| token 级熵（Token-level entropy） | 生成时高熵指示不确定性 [149] | $H(P(x_t)) > \tau$ |
| 序列对数概率（Sequence log-prob） | 输出的平均对数概率低提示编造（confabulation） | $\frac{1}{T} \sum_t \log P(x_t)$ |
| 一致性采样（Consistency sampling） | 生成 $N$ 个回答；一致性低 = 可能是幻觉 [150] | 矛盾率（contradiction rate） |
| 语义熵（Semantic entropy） | 按含义（而非字符串）聚类；高语义熵 = 不确定 [151] | 簇多样性（cluster diversity） |
| DoLA | 对比后层与更早层的 logits；放大事实知识 [152] | 层间发散（layer divergence） |

**语义熵（Semantic Entropy）。** Kuhn 等人 [151] 观察到 token 级熵并不可靠（不同的释义有不同的 token 但含义相同）。他们改为生成多个回答，按语义等价性（通过 NLI）聚类，再在含义簇上计算熵：

$$
SE = -\sum_{c \in \text{clusters}} P(c) \log P(c)
$$

高 $SE$ 意味着模型产出了语义上不同的答案——这是一个强烈的幻觉信号。

**SelfCheckGPT。** Manakul 等人 [150] 通过检查自洽性来检测幻觉：生成多个回答，并验证主回答中的断言是否得到其他备选回答的支持。如果模型"与自己产生分歧"，该断言很可能是幻觉。此方法无需任何外部知识。

**DoLA（按层对比解码，Decoding by Contrasting Layers）。** Chuang 等人 [152] 观察到事实知识涌现于较靠后的 Transformer 层，而较前的层保留更多通用/不确定的表示。DoLA 在每个解码步骤将较后（"成熟"）层与较前（"过早"）层的 logit 分布加以对比：

$$
\text{DoLA}(x_t) = \mathrm{softmax}\!\left(\log P_{\text{late}}(x_t) - \log P_{\text{early}}(x_t)\right)
$$

通过放大深层中所编码的事实知识信号，DoLA 在推理时无需任何重训练即可降低幻觉——只需要对被对比的层额外做一次前向传播。它与基于采样的方法互补，可与之组合使用。

## 1.17 LLM 安全与负责任 AI

安全并非事后补充——它是 LLM 训练流水线不可或缺的一部分。本节介绍 LLM 安全（safety）的关键维度，以及用于约束模型负责任行为的各项机制。

### 1.17.1 威胁分类

表 1.21：LLM 安全威胁类别。

| 类别 | 描述与示例 |
|---|---|
| 有害内容 | 生成有毒、暴力或违法的指令（生物武器、儿童性剥削材料 CSAM） |
| 偏见与歧视 | 固化刻板印象；对不同人群的不公平对待 [153] |
| 隐私侵犯 | 训练数据中的个人身份信息（PII）泄露；记忆提取攻击 [154] |
| 越狱 | 绕过安全护栏的对抗性提示 [155] |
| 虚假信息 | 生成看似可信但虚假的断言（规模化的幻觉） |
| 双重用途 | 将合法能力（编程、化学）武器化用于造成伤害 |

### 1.17.2 安全训练流水线

![图 1.14：安全应用于训练的各个阶段：预训练阶段的数据过滤、SFT 阶段的拒答样本、RLHF 阶段的专用安全奖励模型，以及迭代的红队测试。](images/part-i-foundations/llm-architecture-and-optimization-methods/llm-architecture-and-optimization-methods-p103-13.png)

### 1.17.3 关键安全机制

安全技术：

- **数据过滤（Data filtering）**：从预训练语料中移除有毒、带偏见以及包含 PII 的文本
- **安全 SFT（Safety SFT）**：基于恰当拒答的样本进行训练（「我无法协助此事，因为……」）
- **Constitutional AI [129]**：基于原则进行自我批判；模型依据一套规则「宪法」修订自身的输出
- **安全奖励模型（Safety reward model）**：在带安全标注的样本对上单独训练的奖励模型（RM）；在 RLHF 期间通过加权求和与有用性 RM 组合
- **护栏（Guardrails）**：在服务阶段拦截有害请求/响应的输入/输出分类器
- **红队测试（Red teaming） [156]**：系统化的对抗性评估，用于在部署前发现失败模式

### 1.17.4 有用性—安全性权衡

平衡有用性与安全性：

过度优化安全性会造成**过度拒答（over-refusal）**问题：模型会拒绝良性的请求（例如，在教育情境下拒绝讨论历史上的暴力事件）。目标是获得一个帕累托最优（Pareto-optimal）策略——在安全约束下尽可能有用：

$$
\max_{\theta} \; \mathbb{E}[R_\text{helpful}]
$$

$$
\text{subject to} \quad \mathbb{E}[R_\text{safety}] \geq \tau
$$

在实践中，这通过加权奖励实现：$R = \alpha R_\text{helpful} + (1 - \alpha) R_\text{safety}$，并对 $\alpha$（通常为 0.6–0.8）进行谨慎调校。Meta 的 Llama-3 报告了使用独立的安全与有用性奖励模型，并采用基于边距（margin）的加权方式 [25]。

### 1.17.5 评估

- **安全基准**：ToxiGen、RealToxicityPrompts、BBQ（偏见）、CrowS-Pairs
- **越狱鲁棒性**：GCG 攻击 [155]、多轮越狱、编码化提示
- **过度拒答率**：在良性提示上测量误报式拒答（目标 <5%）
- **红队评估**：由领域专家（生物安全、网络安全）进行的人工对抗性测试

安全永远没有完成之时：

没有任何技术组合能够提供绝对的安全性。新的攻击向量会被不断发现（多模态越狱、移除安全训练的微调攻击、多样本提示）。安全需要持续的监控、对新威胁的快速响应，以及纵深防御（defense-in-depth，多个相互独立的防护层）。
