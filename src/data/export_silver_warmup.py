#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""银标第 5 步：导出预热包（仅负例，不含正例）。"""

from __future__ import annotations

from collections import Counter

from gold_utils import GOLD_OUT, PROJECT_ROOT, iter_jsonl, write_jsonl

IN_PATH = PROJECT_ROOT / "data" / "processed" / "silver" / "silver_filtered.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "silver"
OUT_PATH = OUT_DIR / "silver_warmup.jsonl"


def load_gold_texts() -> set[str]:
    """train/dev/test 任一出现过的句子，银标预热都不要。"""
    texts: set[str] = set()
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        for rec in iter_jsonl(GOLD_OUT / name):
            texts.add(rec["text"])
    return texts


def main() -> None:
    gold_texts = load_gold_texts()
    print(f"gold texts: {len(gold_texts)}")

    kept = []
    n_in = n_pos_skip = n_leak = 0
    neg_type = Counter()

    for rec in iter_jsonl(IN_PATH):
        n_in += 1
        # 正例一律跳过
        if rec.get("spans"):
            n_pos_skip += 1
            continue
        text = rec.get("text") or ""
        if text in gold_texts:
            n_leak += 1
            continue

        nt = rec.get("negative_type") or "hard"
        neg_type[nt] += 1
        kept.append(
            {
                "id": rec["id"].replace("silver_cand", "silver_warm"),
                "text": text,
                "spans": [],
                "source": "silver",
                "status": "accepted",
                "negative_type": nt,
                "split": "train",
                "doc_source": rec.get("doc_source"),
            }
        )

    write_jsonl(OUT_PATH, kept)

    lines = [
        "# Silver Warmup Export (Step 5)",
        "",
        "> 仅负例。正例因抽检估计准确率约 2%，不进入预热包。",
        "",
        f"- filtered in: **{n_in}**",
        f"- skipped positives: **{n_pos_skip}**",
        f"- dropped gold leak: **{n_leak}**",
        f"- warmup kept: **{len(kept)}**",
        f"- hard: **{neg_type.get('hard', 0)}**",
        f"- normal: **{neg_type.get('normal', 0)}**",
        "",
        "参训约定：`source=silver` 只用于单轮预热；主指标仍只看金标 test。",
        "",
    ]
    stats = OUT_DIR / "export_stats.md"
    stats.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(kept)})")
    print(f"Wrote {stats}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
