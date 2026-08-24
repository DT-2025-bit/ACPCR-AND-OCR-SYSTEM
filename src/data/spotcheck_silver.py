# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银标第 4 步：分层抽检样本
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from datetime import date


from gold_utils import PROJECT_ROOT, iter_jsonl

IN_PATH = PROJECT_ROOT / "data" / "processed" / "silver" / "silver_filtered.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "silver"

# 分层 + 抽样函数
def layer_of(rec:dict) -> str:
    # 按记录分成
    if rec.get("spans"):
        return "pos" # 正例
    if rec.get("negative_type") == "hard":
        return "hard" # 难负例
    return "normal" # 普通负例

def sample_layer(items: list, k : int, rng: random.Random) -> list:
    # 每层最多抽 k 条；不够就全取。
    if k >= len(items):
        return list(items)
    return rng.sample(items, k)

def span_brief(rec: dict) -> tuple[str, str, str]:
    """抽出第一条 span 的展示字段；负例为空。"""
    spans = rec.get("spans") or []
    if not spans:
        return"", "", ""
    s = spans[0]
    return (
        f"{s.get('start')}:{s.get('end')}",
        s.get("tongjia", ""),
        s.get("benzi", "")
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pos", type=int, default=150)
    parser.add_argument("--n-pos-tail", type=int, default=50)
    parser.add_argument("--n-hard", type=int, default=150)
    parser.add_argument("--n-normal", type=int, default=50)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    buckets = {"pos": [], "hard": [], "normal": []}
    for rec in iter_jsonl(IN_PATH):
        buckets[layer_of(rec)].append(rec)

    # 正例再按字对频率切:高频 / 长尾
    pair_cnt = Counter()
    pos_head, pos_tail = [], []
    for rec in buckets["pos"]:
        spans = rec.get("spans") or []
        key = (spans[0].get("pair_id") or spans[0].get("tongjia")) if spans else ""
        pair_cnt[key] += 1   # 先计数

    pos_head, pos_tail = [], []
    for rec in buckets["pos"]:
        spans = rec.get("spans") or []
        key = (spans[0].get("pair_id") or spans[0].get("tongjia")) if spans else ""
        if pair_cnt[key] >= 20:
            pos_head.append(rec)
        else:
            pos_tail.append(rec)

    sampled = []
    sampled += sample_layer(pos_head, args.n_pos, rng)
    sampled += sample_layer(pos_tail, args.n_pos_tail, rng)
    sampled += sample_layer(buckets["hard"], args.n_hard, rng)
    sampled += sample_layer(buckets["normal"], args.n_normal, rng)
    rng.shuffle(sampled)

    csv_path = OUT_PATH  / "spotcheck_todo.csv"
    jsonl_path = OUT_PATH  / "spotcheck_todo.jsonl"
    fields = [
        "id", "layer", "text", "pred_span", "pred_tongjia", "pred_benzi",
        "n_spans_before", "negative_type", "doc_source",
        "review_ok", "error_code", "note", "reviewer", "date",
    ]

    rows = []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rec in sampled:
            span_s, tj, bz = span_brief(rec)
            ly = layer_of(rec)
            if ly == "pos":
                spans = rec["spans"]
                key = spans[0].get("pair_id") or spans[0].get("tongjia")
                ly = "pos_head" if pair_cnt[key] >= 20 else "pos_tail"
            row = {
                "id": rec["id"],
                "layer": ly,
                "text": rec["text"],
                "pred_span": span_s,
                "pred_tongjia": tj,
                "pred_benzi": bz,
                "n_spans_before": rec.get("n_spans_before", ""),
                "negative_type": rec.get("negative_type") or "",
                "doc_source": rec.get("doc_source") or "",
                "review_ok": "",  # 你填 1 或 0
                "error_code": "",  # 你填 OK / E_FALSE / E_MISS / E_BENZI / UNCERTAIN
                "note": "",
                "reviewer": "A01",
                "date": str(date.today()),
            }
            w.writerow(row)
            rows.append(row)
    # 同时留一份 jsonl，方便以后算准确率
    with jsonl_path.open("w", encoding="utf-8") as fh:
        import json
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"pool  pos={len(buckets['pos'])} hard={len(buckets['hard'])} normal={len(buckets['normal'])}")
    print(f"pos_head pool={len(pos_head)} pos_tail pool={len(pos_tail)}")
    print(f"sampled: {len(sampled)}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")
    print("请打开 CSV，只填 review_ok / error_code / note 三列。")


if __name__ == "__main__":
    main()


