#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成翻译单元(units)。

- 整章(<= SPLIT_KB):1 个 unit,直接写最终 md。
- 超大章(> SPLIT_KB):按 L3 小节的页码切成多段,每段一个 unit,
  写到 chapters/<part>/<chapter>__segK.md,翻译后由 merge_segments.py 合并。

用法: python3 scripts/generate_units.py frontmatter part-i-foundations part-iii-reasoning
  (参数为 part slug;特殊词 frontmatter 表示包含前置内容)
输出: source/_units_tmp.json(供 workflow 内联)+ 控制台统计
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = json.load(open(os.path.join(ROOT, "source", "_meta.json"), encoding="utf-8"))
SPLIT_KB = 75
SEG_KB = 38


def page_dict(text):
    d, cur, buf = {}, None, []
    for line in text.split("\n"):
        m = re.match(r"<!-- page (\d+) -->", line.strip())
        if m:
            if cur is not None:
                d[cur] = "\n".join(buf)
            cur, buf = int(m.group(1)), []
        else:
            buf.append(line)
    if cur is not None:
        d[cur] = "\n".join(buf)
    return d


def segtext(pd, ps, pe):
    return "\n".join(f"<!-- page {p} -->\n" + pd[p] for p in range(ps, pe + 1) if p in pd)


def page_split(pd, ps, pe, seg_kb):
    """把页码 ps..pe 按文本量切成多段(每段累计 <= seg_kb KB)。"""
    segs, cur, acc = [], ps, 0
    for p in range(ps, pe + 1):
        pl = len(pd.get(p, ""))
        if acc + pl > seg_kb * 1024 and p > cur:
            segs.append((cur, p - 1)); cur, acc = p, 0
        acc += pl
    if cur <= pe:
        segs.append((cur, pe))
    return segs or [(ps, pe)]


def img_page(fn):
    m = re.search(r"-p(\d+)-", fn)
    return int(m.group(1)) if m else 0


def collect_images(part, chapter, out_rel, ps=None, pe=None):
    d = os.path.join(ROOT, "images", part, chapter)
    if not os.path.isdir(d):
        return []
    res = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".png"):
            continue
        pg = img_page(fn)
        if (ps is None) or (ps <= pg <= pe):
            res.append(f"../../images/{part}/{chapter}/{fn}")
    return res


def main():
    args = sys.argv[1:]
    include_fm = "frontmatter" in args
    wanted = [a for a in args if a != "frontmatter"]
    units = []

    def add_single(part, chapter, title, src, out, pages, images):
        units.append({"part": part, "chapter": chapter, "title": title,
                      "segment_source": src, "out": out, "mode": "overwrite",
                      "merge": False, "is_first": True, "pages": pages,
                      "images": images, "context": "整章"})

    def add_split(part, chapter, title, src, out, page_start, page_end, sections, all_imgs):
        src_path = os.path.join(ROOT, src)
        pd = page_dict(open(src_path, encoding="utf-8").read())
        # 叶子单元:超大 L3 按 L4 细化;仍超大者按页码再切,避免单段过大
        leaves = []
        for i, s in enumerate(sections):
            ps = s["page"]
            pe = sections[i + 1]["page"] - 1 if i + 1 < len(sections) else page_end
            subs = s.get("subs") or []
            if len(segtext(pd, ps, pe)) > SEG_KB * 1024 and subs:
                for j, sub in enumerate(subs):
                    leaves.append((sub["title"], sub["page"],
                                   subs[j + 1]["page"] - 1 if j + 1 < len(subs) else pe))
            else:
                leaves.append((s["title"], ps, pe))
        ranges = []
        for (t, sps, spe) in leaves:
            if len(segtext(pd, sps, spe)) > SEG_KB * 1024:
                for (a, b) in page_split(pd, sps, spe, SEG_KB):
                    ranges.append((t, a, b))
            else:
                ranges.append((t, sps, spe))
        # 贪心分组
        groups, g, acc = [], [], 0
        for (st, ps, pe) in ranges:
            slen = len(segtext(pd, ps, pe))
            if g and acc + slen > SEG_KB * 1024:
                groups.append(g); g, acc = [], 0
            g.append((st, ps, pe)); acc += slen
        if g:
            groups.append(g)
        for gi, grp in enumerate(groups):
            ps, pe = grp[0][1], grp[-1][2]
            seg_rel = f"source/{part}/{chapter}__seg{gi+1}.md"
            with open(os.path.join(ROOT, seg_rel), "w", encoding="utf-8") as f:
                f.write(segtext(pd, ps, pe))
            units.append({"part": part, "chapter": chapter, "title": title,
                          "segment_source": seg_rel,
                          "out": f"chapters/{part}/{chapter}__seg{gi+1}.md",
                          "mode": "overwrite", "merge": True,
                          "is_first": (gi == 0), "seg_index": gi + 1,
                          "seg_total": len(groups), "pages": f"p{ps}-p{pe}",
                          "sections": [s[0] for s in grp],
                          "images": collect_images(part, chapter, out, ps, pe),
                          "context": f"第 {gi+1}/{len(groups)} 段"})

    if include_fm:
        for fm in META["frontmatter"]:
            src = f"source/frontmatter/{fm['slug']}.md"
            add_single("frontmatter", fm["slug"], fm["title"], src, fm["file"],
                       f"p{fm['page_start']}-p{fm['page_end']}",
                       collect_images("frontmatter", fm["slug"], fm["file"]))
    for p in META["parts"]:
        if p["slug"] not in wanted:
            continue
        for c in p["chapters"]:
            src = f"source/{p['slug']}/{c['slug']}.md"
            out = c["file"]
            size = os.path.getsize(os.path.join(ROOT, src))
            imgs = collect_images(p["slug"], c["slug"], out)
            if size > SPLIT_KB * 1024 and c["sections"]:
                add_split(p["slug"], c["slug"], c["title"], src, out,
                          c["page_start"], c["page_end"], c["sections"], imgs)
            else:
                add_single(p["slug"], c["slug"], c["title"], src, out,
                           f"p{c['page_start']}-p{c['page_end']}", imgs)

    json.dump(units, open(os.path.join(ROOT, "source/_units_tmp.json"), "w"),
              ensure_ascii=False, indent=2)
    n_split = len({u["chapter"] for u in units if u["merge"]})
    print(f"✅ 生成 {len(units)} 个翻译单元,涉及 {len({u['chapter'] for u in units})} 章/篇,其中 {n_split} 章拆分")
    for u in units:
        flag = "拆" if u["merge"] else "整"
        print(f"  [{flag}] {u['part']}/{u['chapter'][:40]:40} {u['pages']:>10} | {u['context']}")


if __name__ == "__main__":
    main()
