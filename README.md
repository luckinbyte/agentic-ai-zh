# Agentic AI 指南 —— 从基础到系统(中文翻译版)

> 📖 **在线阅读**:[https://luckinbyte.github.io/agentic-ai-zh/](https://luckinbyte.github.io/agentic-ai-zh/)

> 中文翻译自 _The Hitchhiker's Guide to Agentic AI: From Foundations to Systems_
> 原作者:Haggai Roitman · 2026 · v1.2.2 · [arXiv:2606.24937](https://arxiv.org/abs/2606.24937)

本仓库是将该英文专著翻译为**简体中文**的 mdBook 电子书项目,按章节组织为多个 Markdown 文件。

## 翻译约定

- 正文译为简体中文,意译为主;专业术语首次出现标注英文原词。
- 通用缩写保留英文:RAG / MCP / PPO / GRPO / LoRA / SFT / RLHF / Transformer / attention 等。
- 数学公式用 LaTeX(`$...$` / `$$...$$`)。
- 代码保留英文原文,关键处加中文注释。
- 插图提取并引用,附中文图注。

术语对照见 [GLOSSARY.md](GLOSSARY.md),翻译规范见 [TRANSLATION-GUIDE.md](TRANSLATION-GUIDE.md)。

## 目录结构

```
book.toml             mdBook 配置(主题、折叠侧栏、搜索、KaTeX 公式)
src/                  书籍源码(mdBook 只构建此目录)
  SUMMARY.md          mdBook 目录(前言 + 29 章,按 Part 分组)
  chapters/           译文(按 Part 分目录)
    frontmatter/      免责声明、作者简介、前言、引言
    part-i-foundations/ ... part-vi-assessment-reference/
  images/             提取的插图
GLOSSARY.md           中英术语对照表
TRANSLATION-GUIDE.md  翻译规范
source/               预处理切出的原文分章文本(便于追溯,不入电子书)
```

## 本地预览

需要 [mdBook](https://rust-lang.github.io/mdBook/) 与 [mdbook-katex](https://github.com/lzanini/mdbook-katex)(用于渲染 LaTeX 公式):

```bash
cargo install mdbook --version 0.4.48 --locked
cargo install mdbook-katex --version 0.9.4 --locked
mdbook serve --open     # 本地实时预览 http://localhost:3000
```

> 注:mdbook-katex 0.9.4 与 mdBook 0.5.x 不兼容,需固定使用 0.4.48。

> 本译作仅供学习研究;版权归原作者所有。
