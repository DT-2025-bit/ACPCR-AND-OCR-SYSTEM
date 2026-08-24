# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银标第 2 步: 规范表字头匹配 → 生成候选 span（未过滤）。

"""
from __future__ import annotations

import re
import json
import argparse  # 命令行参数，如 --max-sents 20000
from collections import defaultdict, Counter
from pathlib import Path

from gold_utils import GOLD_OUT, PROJECT_ROOT, iter_jsonl, write_jsonl

SILVER_CORPUS = PROJECT_ROOT / "data" / "Silver Label Data" / "output" / "corpus.jsonl" #银标正文
LEXICON_PATH = GOLD_OUT / "lexicon.jsonl" #规范表
GOLD_TEST = GOLD_OUT / "test.jsonl" #金标 test 全文
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "silver" #输出目录

SENT_SPLIT_RE = re.compile(r"[。！？；!\?]+")

def split_sentences(text: str) -> list[str]:
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def load_heads(lexicon_path: Path) -> dict[str, list[str]]: 
    """加载规范表字头 → 本字列表（一通多本都留下）。"""
    heads: dict[str, list[str]] = defaultdict(list)
    for row in iter_jsonl(lexicon_path):
        tj = row["tongjia"]
        bz = row["benzi"]
        if bz not in heads[tj]:
            heads[tj].append(bz)
    return dict(heads)


def load_gold_test_texts(path: Path) -> set[str]:
    """金标 test 全文集合，用来防泄露。"""
    return {row["text"] for row in iter_jsonl(path)}


def match_spans(text: str, heads: dict[str, list[str]]) -> list[dict]:
    """扫描每个字符：若是规范表通假字头，就生成候选 span。"""
    spans = []
    for i, ch in enumerate(text):
        if ch not in heads:
            continue
        benzis = heads[ch]
        spans.append(
            {
                "start": i,
                "end": i + 1,
                "type": "tongjia",
                "tongjia": ch,
                "benzi": benzis[0],  # 先取第一个；完整列表放下面
                "benzi_candidates": benzis,
                "confidence": "low",
                "pair_id": f"TJ_{ch}_{benzis[0]}",
            }
        )
    return spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=200, help="先抽样，确认逻辑再加大")
    parser.add_argument("--max-sents", type=int, default=20000)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args() 
    OUT_DIR.mkdir(parents=True, exist_ok=True) # 确保文件存在
    print("Loading lexicon heads...")
    heads = load_heads(LEXICON_PATH) # 加载规范表字典
    print(f"  heads: {len(heads)}")
    print("Loading gold test texts...")
    gold_test = load_gold_test_texts(GOLD_TEST) # 加载金标 test 全文
    print(f"  gold test texts: {len(gold_test)}")
    records = [] # 结果
    n_docs = 0 # 篇数
    n_sents = 0 # 句子数
    n_leak_drop = 0 # 泄露丢弃数
    n_pos = 0 # 有 span 的句子数
    spans_per_sent = []
    head_hits = Counter()
    with SILVER_CORPUS.open(encoding="utf-8") as fh: # 遍历银标正文
        for line in fh:
            if n_docs >= args.max_docs or n_sents >= args.max_sents:
                break
            doc = json.loads(line)
            n_docs += 1 # 更新篇数
            content = doc.get("content") or ""
            for sent in split_sentences(content): # 分割篇数
                if n_sents >= args.max_sents:
                    break
                # 去掉空白，按字符长度过滤
                text = re.sub(r"\s+", "", sent)
                if not (args.min_len <= len(text) <= args.max_len):
                    continue
                # 防泄露：与金标 test 撞车则丢弃
                if text in gold_test:
                    n_leak_drop += 1
                    continue
                spans = match_spans(text, heads)
                n_sents += 1 # 更新句子数
                spans_per_sent.append(len(spans))
                if spans:
                    n_pos += 1 # 更新有 span 的句子数
                    for s in spans:
                        head_hits[s["tongjia"]] += 1
                records.append(
                    {
                        "id": f"silver_cand_{n_sents:06d}",
                        "text": text,
                        "spans": spans,
                        "source": "silver",
                        "status": "candidate",  # 未过滤
                        "doc_id": doc.get("id"),
                        "doc_source": doc.get("source"),
                    }
                )
    out_path = OUT_DIR / "candidates_sample.jsonl"
    write_jsonl(out_path, records)

    # ---- 统计 ----
    spans_per_sent.sort()

    def pct(vals: list[int], p: float) -> int:
        if not vals:
            return 0
        i = min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))
        return vals[i] #算 P50/P90（一半句子 ≤ 多少个 span） 

    avg = sum(spans_per_sent) / max(len(spans_per_sent), 1)
    lines = [
        "# Silver Candidate Match Stats (Step 2)",
        "",
        f"- docs scanned: **{n_docs}**",
        f"- sentences kept: **{n_sents}**",
        f"- dropped (gold test leak): **{n_leak_drop}**",
        f"- sentences with ≥1 span: **{n_pos}** ({100 * n_pos / max(n_sents, 1):.1f}%)",
        f"- spans/sent avg: **{avg:.2f}**",
        f"- spans/sent P50/P90/max: **{pct(spans_per_sent, 50)} / {pct(spans_per_sent, 90)} / {spans_per_sent[-1] if spans_per_sent else 0}**",
        "",
        "## Top-20 heads",
        "",
    ]
    for ch, c in head_hits.most_common(20):
        lines.append(f"- `{ch}`: {c}")

    stats_path = OUT_DIR / "match_stats.md"
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_path} ({len(records)} records)")
    print(f"Wrote {stats_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
