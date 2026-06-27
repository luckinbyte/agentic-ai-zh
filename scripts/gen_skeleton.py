#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段①:基于 PDF 的 TOC 生成 GitBook 骨架。

输出:
  - SUMMARY.md                      GitBook 目录(标题暂用英文,翻译后由脚本更新为中文)
  - chapters/<part-slug>/<chapter-slug>.md   各章骨架(H1=章 / H2=小节 / H3=子节)
  - chapters/frontmatter/<slug>.md           前置内容骨架
  - source/_meta.json               全书章节元数据(页码范围等,供预处理与翻译使用)
"""
import json
import re
import os
import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOC_PATH = os.path.join(ROOT, "source", "_toc.json")
PDF_PATH = os.path.join(ROOT, "2606.24937.pdf")
CHAPTERS_DIR = os.path.join(ROOT, "chapters")
META_PATH = os.path.join(ROOT, "source", "_meta.json")
SUMMARY_PATH = os.path.join(ROOT, "SUMMARY.md")

VALID_ROMANS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}
ROMAN_RE = re.compile(r"^([IVXLC]+)\s+(.+)$")


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def main():
    toc = json.load(open(TOC_PATH, encoding="utf-8"))
    doc = fitz.open(PDF_PATH)
    page_count = doc.page_count

    frontmatter = []          # L1 条目(在首个 Part 之前)
    parts = []                # Part 列表
    chapters_all = []         # 全书所有 L2 章(扁平,有序)
    current_part = None
    seen_part = False

    for e in toc:
        lvl, title, page = e["level"], e["title"].strip(), e["page"]
        m = ROMAN_RE.match(title)
        is_part = lvl == 1 and m and m.group(1) in VALID_ROMANS
        if is_part:
            seen_part = True
            roman = m.group(1)
            name = m.group(2).strip()
            current_part = {
                "roman": roman, "name": name,
                "slug": f"part-{roman.lower()}-{slugify(name)}",
                "page": page, "chapters": [],
            }
            parts.append(current_part)
        elif lvl == 1 and not seen_part:
            frontmatter.append({"title": title, "page": page, "slug": slugify(title)})
        elif lvl == 2 and current_part is not None:
            ch = {
                "title": title, "page": page, "slug": slugify(title),
                "part": current_part["slug"], "part_roman": current_part["roman"],
                "index": len(current_part["chapters"]) + 1,
                "sections": [],
            }
            current_part["chapters"].append(ch)
            chapters_all.append(ch)
        elif lvl == 3 and chapters_all:
            chapters_all[-1]["sections"].append({"title": title, "page": page, "subs": []})
        elif lvl == 4 and chapters_all and chapters_all[-1]["sections"]:
            chapters_all[-1]["sections"][-1]["subs"].append({"title": title, "page": page})

    # 边界页 = 所有 L1 + L2 页,用于推断每条目的内容页范围
    boundaries = sorted({e["page"] for e in toc if e["level"] in (1, 2)})

    def range_end(page: int) -> int:
        after = [b for b in boundaries if b > page]
        return (min(after) - 1) if after else page_count

    # 写 frontmatter 骨架
    fm_meta = []
    os.makedirs(os.path.join(CHAPTERS_DIR, "frontmatter"), exist_ok=True)
    for fm in frontmatter:
        ps, pe = fm["page"], range_end(fm["page"])
        fm["page_start"], fm["page_end"] = ps, pe
        path = os.path.join(CHAPTERS_DIR, "frontmatter", f"{fm['slug']}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {fm['title']}\n\n")
            f.write(f"> 本节待翻译(原文页码:p{ps}–p{pe})。\n")
        fm_meta.append({"slug": fm["slug"], "title": fm["title"],
                        "page_start": ps, "page_end": pe,
                        "file": f"chapters/frontmatter/{fm['slug']}.md"})

    # 写各章骨架 + 收集元数据
    for part in parts:
        part_dir = os.path.join(CHAPTERS_DIR, part["slug"])
        os.makedirs(part_dir, exist_ok=True)
        for ch in part["chapters"]:
            ps, pe = ch["page"], range_end(ch["page"])
            ch["page_start"], ch["page_end"] = ps, pe
            ch["file"] = f"chapters/{part['slug']}/{ch['slug']}.md"
            path = os.path.join(ROOT, ch["file"])
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {ch['title']}\n\n")
                f.write(f"> 本章待翻译(原文页码:p{ps}–p{pe})。\n\n")
                for sec in ch["sections"]:
                    f.write(f"\n## {sec['title']}\n\n")
                    for sub in sec["subs"]:
                        f.write(f"\n### {sub['title']}\n\n")

    # 元数据
    meta = {
        "page_count": page_count,
        "frontmatter": fm_meta,
        "parts": [
            {
                "roman": p["roman"], "name": p["name"], "slug": p["slug"],
                "page": p["page"],
                "chapters": [
                    {
                        "index": c["index"], "slug": c["slug"], "title": c["title"],
                        "part": c["part"], "part_roman": c["part_roman"],
                        "page_start": c["page_start"], "page_end": c["page_end"],
                        "file": c["file"],
                        "sections": c["sections"],
                    } for c in p["chapters"]
                ],
            } for p in parts
        ],
    }
    json.dump(meta, open(META_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # SUMMARY.md
    lines = ["# Summary", "",]
    lines.append("## 前言")
    for fm in fm_meta:
        lines.append(f"* [{fm['title']}]({fm['file']})")
    lines.append("")
    for p in meta["parts"]:
        lines.append(f"## Part {p['roman']}　{p['name']}")
        for c in p["chapters"]:
            lines.append(f"* [{c['title']}]({c['file']})")
        lines.append("")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 汇报
    n_ch = sum(len(p["chapters"]) for p in parts)
    n_sec = sum(len(c["sections"]) for p in parts for c in p["chapters"])
    n_sub = sum(len(s["subs"]) for p in parts for c in p["chapters"] for s in c["sections"])
    print(f"✅ 骨架生成完成")
    print(f"   Frontmatter: {len(fm_meta)} 篇")
    print(f"   Parts: {len(parts)} | Chapters: {n_ch} | Sections: {n_sec} | Subsections: {n_sub}")
    print(f"   SUMMARY.md, source/_meta.json 已生成")


if __name__ == "__main__":
    main()
