# 《The Hitchhiker's Guide to Agentic AI》中文翻译电子书 — 设计文档

- **日期**:2026-06-27
- **原文**:《The Hitchhiker's Guide to Agentic AI: From Foundations to Systems》,Haggai Roitman, 2026, v1.2.2(arXiv:2606.24937)
- **规模**:603 页 / 6 Parts / 29 章(L2)/ 277 小节(L3)/ 668 子节(L4)
- **当前目录**:`/root/gitbook/agentic`(全新,非 git 仓库)

## 1. 目标

将英文专著翻译为简体中文 GitBook 电子书,按章节拆分为多个 Markdown 文件,后续提交到 GitHub。

## 2. 关键决策(已与用户确认)

| 维度 | 决策 |
|---|---|
| 翻译范围 | **全书分阶段推进**:先搭全书骨架,首批翻译 Part V「Agentic AI」,review 后再扩展其余 Part |
| 文件粒度 | **每章(L2)= 1 个 md 文件**,按 Part 分目录;章内小节用标题层级 |
| 翻译风格 | **简体中文正文**,意译为主;术语首现「中文(English)」;通用缩写保留英文(RAG/MCP/PPO/GRPO/LoRA/SFT/RLHF/Transformer/attention) |
| 非文字元素 | **完整保留**:提取图片并引用;公式转 LaTeX(`$...$`/`$$...$$`);代码保留英文原文+关键处中文注释 |
| 推进方式 | **Workflow 多 agent 并行**,分阶段;首批 Part V |

## 3. 仓库结构

```
agentic/
├── README.md                # 项目说明(中英)
├── SUMMARY.md               # GitBook 目录
├── book.json                # GitBook 配置
├── GLOSSARY.md              # 中英术语对照表(全书统一,agent 共享)
├── TRANSLATION-GUIDE.md     # 翻译规范(喂给每个 agent)
├── .gitignore
├── images/                  # 提取的插图,按 part/chapter 组织
├── source/                  # 预处理切出的原文分章文本(中间产物,默认入库便于追溯)
└── chapters/                # 译文,按 Part 分目录
    ├── frontmatter/         # 免责声明、作者简介、前言、引言
    ├── part-i-foundations/
    ├── part-v-agentic-ai/
    └── part-vi-reference/
```

- 文件名:英文 kebab-case(GitBook URL 友好);文件内标题用中文。
- Part 目录命名:`part-i-foundations` … `part-v-agentic-ai` … `part-vi-reference`。

## 4. 三阶段流水线

### 阶段 ① 骨架生成(Python,读 PDF TOC)
- 读取 pymupdf 的 984 条 TOC。
- 生成 `SUMMARY.md`(GitBook 格式)。
- 生成 29 章文件,内含 L3/L4 标题骨架(H2/H3)。
- frontmatter(Disclaimer / About the Author / Preface / Introduction)单独文件。

### 阶段 ② 预处理(Python + pymupdf)
- 按每章页码范围,把原文文本切到 `source/<part>/<chapter>.md`(保留结构标记)。
- 把该章页面插图提取到 `images/<part>/<chapter>/`,命名按出现顺序。
- 理由:agent 不便直接读 PDF,先存干净文本交给 agent。

### 阶段 ③ 并行翻译(Workflow)
- 每个 agent 输入:该章原文文本 + `TRANSLATION-GUIDE.md` + `GLOSSARY.md`。
- 输出:中文 markdown 到 `chapters/<part>/<chapter>.md`。
- 每批一个 Workflow;首批 Part V(12 章)。
- 翻译后可选一致性 review agent。

## 5. 翻译规范要点(`TRANSLATION-GUIDE.md`)

- 简体中文,意译为主,保技术准确。
- 术语首现「中文(English)」;通用缩写保留英文。
- 公式 → LaTeX;代码保留原文+关键中文注释;图片 `![中文图注](images/...)` + 图注。
- 人名、参考文献、URL 保留原文。
- 保留 markdown 结构:标题、列表、表格、引用。

## 6. 质量保证

- `GLOSSARY.md` 统一术语,首批 Part V 沉淀,全书沿用。
- 每章译后脚本检查:标题层级完整性、图片引用有效性、明显未译残留英文。
- 每批结束 review checkpoint。

## 7. 首批交付物

全书骨架(SUMMARY + 29 章标题骨架)+ **Part V 完整 12 章中文译文** + 图片提取 + 术语表初稿 + 翻译规范文档 + `git init` 与 README/.gitignore。

## 8. Part / Chapter 全景(取自 TOC)

- Frontmatter:Disclaimer(p24), About the Author(p25), Preface(p26), Introduction(p30)
- **I Foundations**(p34): LLM Architecture and Optimization Methods(p35), Systems Foundations for LLMs(p105), Introduction to Reinforcement Learning(p119)
- **II RL Methods for LLMs**(p132): RL Foundations(p133), PPO(p136), DPO(p145), GRPO(p158), Preference Optimization Variants(p174), Reward Model Training(p182), SFT Best Practices(p190), System Architecture & Infrastructure at Scale(p199), LLM Agentic Training(p222)
- **III Reasoning**(p250): RL for Large Reasoning Models(p251)
- **IV Evaluation**(p273): LLM Evaluation(p274)
- **V Agentic AI**(p292): Introduction to Agentic AI(p292), RAG(p295), Agentic Memory Systems(p320), Agent Harness(p343), Agent Design Patterns(p369), Agentic Environments and Benchmarks(p375), MCP(p392), Agent Skills(p412), A2A(p417), Multi-Agent Systems(p439), Agent Development Frameworks(p459), Agentic UI Frameworks(p490)
- **VI Assessment & Reference**(p512): Quiz Questions & Detailed Answers(p513), Quick Reference(p565), Conclusion and Future Directions(p573)

## 9. 风险与注意

- 公式从 PDF 提取常为 Unicode/乱码,需 agent 据上下文重建 LaTeX,可能不完美。
- 图片与正文位置对应靠页码范围近似,个别图归属可能需人工微调。
- 全书工作量大,必须分批;术语表是全书一致性的关键。
