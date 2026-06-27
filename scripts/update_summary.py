#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从各章译文的一级标题(H1)提取中文标题,回填到 SUMMARY.md 对应条目。

仅更新已有译文的章节(文件首行为 `# 中文标题`);未翻译的保留英文标题。
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

meta = json.load(open("source/_meta.json", encoding="utf-8"))


def first_h1(rel_path):
    path = rel_path
    if not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8").read().lstrip().splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            # 跳过仍是骨架占位的英文标题(简单启发:若与原英文标题相同则不替换)
            return t
    return None


title_map = {}
for fm in meta["frontmatter"]:
    t = first_h1(fm["file"])
    if t:
        title_map[fm["file"]] = t
for p in meta["parts"]:
    for c in p["chapters"]:
        t = first_h1(c["file"])
        if t:
            title_map[c["file"]] = t

lines = open("SUMMARY.md", encoding="utf-8").read().splitlines()
out = []
replaced = 0
for line in lines:
    m = re.match(r"(\*\s*)\[([^\]]*)\]\(([^)]+)\)", line)
    if m and m.group(3) in title_map:
        out.append(f"{m.group(1)}[{title_map[m.group(3)]}]({m.group(3)})")
        replaced += 1
    else:
        out.append(line)
open("SUMMARY.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"✅ SUMMARY.md 已更新,{replaced} 个条目替换为中文标题")
