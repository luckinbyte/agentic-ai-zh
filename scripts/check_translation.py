#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""质量检查:扫描某 Part 译文,报告字符数、中文占比、图片引用有效性、骨架残留等。

用法: python3 scripts/check_translation.py chapters/part-v-agentic-ai
"""
import os
import re
import sys


def check(d):
    issues = []
    print(f"{'文件':<46}{'字符':>7}{'中文':>7}{'图片':>5}  状态")
    print("-" * 80)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(d, fn)
        txt = open(path, encoding="utf-8").read()
        chars = len(txt)
        cjk = sum(1 for c in txt if "一" <= c <= "鿿")
        n_img = len(re.findall(r"!\[", txt))
        flags = []
        if ("本章待翻译" in txt) or ("本节待翻译" in txt):
            flags.append("骨架残留")
        if not txt.lstrip().startswith("# "):
            flags.append("缺H1")
        if chars < 200:
            flags.append("过短")
        if chars > 200 and cjk / max(chars, 1) < 0.25:
            flags.append("中文占比低")
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", txt):
            imgp = os.path.normpath(os.path.join(d, m.group(1)))
            if not os.path.exists(imgp):
                flags.append(f"图片失效:{m.group(1)}")
        status = "✅" if not flags else "⚠ " + ";".join(flags)
        print(f"{fn:<46}{chars:>7}{cjk:>7}{n_img:>5}  {status}")
        for f in flags:
            issues.append((fn, f))
    print("-" * 80)
    print(f"问题数: {len(issues)}")
    return issues


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "chapters/part-v-agentic-ai"
    check(d)
