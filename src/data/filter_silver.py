#!/usr/bin/env python3
# -*- coding:utf - 8 -*-
"""银标第 3 步：对候选 span 做规则过滤。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from gold_utils import GOLD_OUT, PROJECT_ROOT, iter_jsonl, write_jsonl

CAND_PATH = PROJECT_ROOT / "data" / "processed" / "silver" / "candidates_sample.jsonl"
GOLD_TRAIN = GOLD_OUT / "train.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "silver"

# 加入停止词表
STOP_HEADS = set(
    "以不有于人中而者大十年王后文事至平高司史"
    "是也乃遂言正同北"
    "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
)
MAX_SPANS_KEEP = 3
MAX_SPANS_POSITIVE = 1

# 读金标 train 的字对
def load_gold_pairs(path: Path) -> set[tuple[str, str]]:
    """只用train, 禁止用 test 调规则。"""
    pairs: set[tuple[str, str]] = set()
    for rec in iter_jsonl(path):
        for span in rec.get("spans") or []:
            tj = span.get("tongjia")
            bz = span.get("benzi")
            if tj and bz:
                pairs.add((tj, bz))
    return pairs

# 过滤一条样本
def filter_spans(rec: dict, gold_pairs: set[tuple[str, str]]) -> dict | None:
    old_spans = rec.get("spans") or []
    kept = []
    # 过滤span
    for span in old_spans:
        tj = span.get("tongjia", "")
        bz = span.get("benzi", "")
        if tj in STOP_HEADS:
            continue
        if (tj, bz) not in gold_pairs:
            continue
        kept.append(span)

    too_noisy = len(kept) > MAX_SPANS_KEEP
    if too_noisy:
        kept = []  # 筛完还剩 超过 3 个 span，整句当噪声，span 全清空
    elif kept:
        kept = kept[:MAX_SPANS_POSITIVE]  # 正例每句只留 1 个

    out = dict(rec) # 浅拷贝第 2 步的候选记录
    out["spans"] = kept
    out["source"] = "silver"
    out["n_spans_before"] = len(old_spans)

    if kept:
        out["status"] = "accepted"
        out["negative_type"] = None # 正例
    elif old_spans:
        out["status"] = "accepted"
        out["negative_type"] = "hard" # 难负例
    else:
        out["status"] = "accepted"
        out["negative_type"] = "normal" # 正常负例
    return out

# main（读候选 → 过滤 → 写统计）
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-record", type=int,default=20000)
    args = parser.parse_args()

    print("Loading gold train pairs...")
    gold_pairs = load_gold_pairs(GOLD_TRAIN) # 加载金标 train 的字对
    print(f"  pairs: {len(gold_pairs)}")

    kept_records = [] # 过滤后的记录
    n_in = 0 # 输入的记录数
    n_drop_noisy = 0 # 过滤后丢弃的记录数
    n_pos = n_hard = n_normal = 0 # 正例、难负例、正常负例的记录数
    spans_after = [] # 过滤后保留的span数
    head_hits = Counter() # 字头命中数

    for rec in iter_jsonl(CAND_PATH):
        if n_in >= args.max_record:
            break
        n_in += 1
        out = filter_spans(rec, gold_pairs)
        kept_records.append(out)
        if out.get("n_spans_before", 0) > 0 and not out["spans"] and out.get("negative_type") == "hard":
            n_drop_noisy += 1  # 这里改意为：被压成难负例的句数
        spans_after.append(len(out["spans"]))
        if out["spans"]:
            n_pos += 1
            for s in out["spans"]:
                head_hits[s["tongjia"]] += 1
        elif out.get("negative_type") == "hard":
            n_hard += 1
        else:
            n_normal += 1

    out_path = OUT_DIR / "silver_filtered.jsonl"
    write_jsonl(out_path, kept_records)

    n_kept = len(kept_records) or 1
    lines = [
        "# Silver Filter Stats (Step 3)",
        "",
        f"- candidates in: **{n_in}**",
        f"- dropped (too many spans after filter): **{n_drop_noisy}**",
        f"- kept: **{len(kept_records)}**",
        f"- positive: **{n_pos}** ({100 * n_pos / n_kept:.1f}%)",
        f"- hard negative: **{n_hard}** ({100 * n_hard / n_kept:.1f}%)",
        f"- normal negative: **{n_normal}** ({100 * n_normal / n_kept:.1f}%)",
        "",
        "## Top-20 heads after filter",
        "",
    ]
    for ch, c in head_hits.most_common(20):
        lines.append(f"- `{ch}`: {c}")

    stats_path = OUT_DIR / "filter_stats.md"
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {stats_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

