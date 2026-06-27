#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并超大章的 __segK.md 片段为最终章节文件。

对每个存在 __seg 片段的章节:按 seg 序号排序拼接;
首段保留全部(含 # 章标题),后续段去掉顶层 "# " 标题(避免重复 H1);
写回 chapters/<part>/<chapter>.md,删除片段文件。
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta = json.load(open(os.path.join(ROOT, "source", "_meta.json"), encoding="utf-8"))

merged = 0
for p in meta["parts"]:
    for c in p["chapters"]:
        out = os.path.join(ROOT, c["file"])
        part_dir = os.path.dirname(out)
        segs = glob.glob(os.path.join(part_dir, c["slug"] + "__seg*.md"))
        if not segs:
            continue

        def idx(f):
            m = re.search(r"__seg(\d+)\.md$", f)
            return int(m.group(1)) if m else 0
        segs.sort(key=idx)

        chunks = []
        for i, f in enumerate(segs):
            txt = open(f, encoding="utf-8").read().strip()
            if i > 0:
                cleaned = [ln for ln in txt.split("\n")
                           if not ln.startswith("# ") or ln.startswith("## ")]
                txt = "\n".join(cleaned).strip()
            chunks.append(txt)
        with open(out, "w", encoding="utf-8") as fo:
            fo.write("\n\n".join(chunks).strip() + "\n")
        for f in segs:
            os.remove(f)
        merged += 1
        print(f"  合并 {c['slug']}: {len(segs)} 段 -> {os.path.relpath(out, ROOT)}")

print(f"✅ 合并 {merged} 章")
