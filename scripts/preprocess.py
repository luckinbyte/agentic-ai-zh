#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段②:按章页码范围,从 PDF 提取原文文本与图片。

对每个 frontmatter / chapter:
  - 原文文本 -> source/<part-slug>/<chapter-slug>.md(含 <!-- page N --> 标记)
  - 插图     -> images/<part-slug>/<chapter-slug>/img-p<page>-<idx>.png
                (按显示区域 clip 渲染,过滤 <60pt 的小图标)

用法:
  python3 scripts/preprocess.py            # 全书
  python3 scripts/preprocess.py part-v-agentic-ai   # 仅指定 Part
"""
import json
import os
import sys
import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(ROOT, "2606.24937.pdf")
META_PATH = os.path.join(ROOT, "source", "_meta.json")
SOURCE_DIR = os.path.join(ROOT, "source")
IMAGES_DIR = os.path.join(ROOT, "images")

MIN_IMG_PT = 60.0      # 过滤小图标
DPI = 150.0            # 图片渲染分辨率


def extract_range(doc, ps, pe, out_text, img_dir, img_prefix):
    os.makedirs(os.path.dirname(out_text), exist_ok=True)
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
    parts = []
    img_count = 0
    for pno in range(ps - 1, pe):          # 页码 ps..pe(闭区间)
        page = doc[pno]
        parts.append(f"\n\n<!-- page {pno + 1} -->\n")
        parts.append(page.get_text("text"))
        if img_dir:
            for info in page.get_image_info():
                x0, y0, x1, y1 = info["bbox"]
                w, h = x1 - x0, y1 - y0
                if w < MIN_IMG_PT or h < MIN_IMG_PT:
                    continue
                try:
                    pix = page.get_pixmap(clip=fitz.Rect(x0, y0, x1, y1),
                                          matrix=fitz.Matrix(DPI / 72.0, DPI / 72.0))
                    img_count += 1
                    fn = f"{img_prefix}-p{pno + 1}-{img_count:02d}.png"
                    pix.save(os.path.join(img_dir, fn))
                except Exception as e:
                    print(f"   ⚠ 图片渲染失败 p{pno+1}: {e}")
    with open(out_text, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return img_count


def main():
    only_part = sys.argv[1] if len(sys.argv) > 1 else None
    meta = json.load(open(META_PATH, encoding="utf-8"))
    doc = fitz.open(PDF_PATH)

    total_imgs = 0
    # frontmatter
    if not only_part:
        for fm in meta["frontmatter"]:
            out = os.path.join(SOURCE_DIR, "frontmatter", f"{fm['slug']}.md")
            n = extract_range(doc, fm["page_start"], fm["page_end"], out, None, fm["slug"])
            print(f"  frontmatter/{fm['slug']}: p{fm['page_start']}-{fm['page_end']}")

    for part in meta["parts"]:
        if only_part and part["slug"] != only_part:
            continue
        for ch in part["chapters"]:
            out = os.path.join(SOURCE_DIR, part["slug"], f"{ch['slug']}.md")
            img_dir = os.path.join(IMAGES_DIR, part["slug"], ch["slug"])
            n = extract_range(doc, ch["page_start"], ch["page_end"], out, img_dir, ch["slug"])
            total_imgs += n
            size_kb = os.path.getsize(out) // 1024
            print(f"  {part['slug']}/{ch['slug']}: p{ch['page_start']}-{ch['page_end']} | {size_kb}KB 文本 | {n} 图")

    print(f"\n✅ 预处理完成,共提取 {total_imgs} 张图片")


if __name__ == "__main__":
    main()
