#!/usr/bin/env python3
"""银标第 1 步：探索性分析（EDA）

银标原料 ≠ 已标注通假。
Silver Label Data/output/corpus.jsonl 是古典文献正文（篇章级），
还没有通假 span。银标要靠「规范表匹配 + 规则」自动打出来。

"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from gold_utils import GOLD_OUT, PROJECT_ROOT, iter_jsonl, strip_head_suffix

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
SILVER_CORPUS = PROJECT_ROOT / "data" / "Silver Label Data" / "output" / "corpus.jsonl"
LEXICON_PATH = GOLD_OUT / "lexicon.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "silver"
REPORT_PATH = OUT_DIR / "eda_report.md"

# 切句子
SENT_SPLIT_RE = re.compile(r"[。！？；!\?]+")


def load_tongjia_heads(lexicon_path: Path) -> set[str]:
    """从规范表取出所有「可作为通假字」的字头集合。"""
    heads: set[str] = set()
    for row in iter_jsonl(lexicon_path):
        heads.add(strip_head_suffix(row["tongjia"]))
    return heads


def split_sentences(text: str) -> list[str]:
    """粗切句：按句读符号切开，去掉空白和过短碎片。"""
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def percentile(sorted_vals: list[int], p: float) -> int:
    """句子长度分布百分位数"""
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(round((p / 100) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True) # 创建输出目录
    print("Loading lexicon heads...", flush=True) 
    heads = load_tongjia_heads(LEXICON_PATH) # 加载规范表字头
    print(f"  tongjia heads in lexicon: {len(heads):,}")

    print("Scanning silver corpus...", flush=True)
    n_docs = 0
    by_source: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    total_chars = 0

    # 句级统计（对超长篇章也切句）
    sent_lens: list[int] = []
    n_sents = 0
    n_sents_with_head = 0  # 句中至少命中 1 个规范表通假字头
    head_hit_counts: Counter[str] = Counter()

    # 为控制耗时：每篇最多统计前500句做字头命中（全文仍计入篇章统计）
    MAX_SENTS_PER_DOC = 500

    with SILVER_CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            n_docs += 1
            content = rec.get("content") or ""
            total_chars += len(content)
            by_source[rec.get("source", "?")] += 1
            by_category[rec.get("category", "?")] += 1

            sents = split_sentences(content) # 切句子
            for i, sent in enumerate(sents):
                if i >= MAX_SENTS_PER_DOC:
                    break
                # 去掉空白后测长度（近似「字符数」）
                compact = re.sub(r"\s+", "", sent)
                if len(compact) < 4:
                    continue
                n_sents += 1
                sent_lens.append(len(compact))

                # 远监督信号：句中是否出现规范表通假字头
                hit = False
                for ch in set(compact):
                    if ch in heads:
                        hit = True
                        head_hit_counts[ch] += 1
                if hit:
                    n_sents_with_head += 1

    sent_lens.sort()
    hit_rate = 100 * n_sents_with_head / max(n_sents, 1)

    # ---- 控制台摘要 ----
    print(f"\nDocuments: {n_docs:,}")
    print(f"Total chars: {total_chars:,}")
    print(f"Sentences scored: {n_sents:,}")
    print(
        f"Sent len P50/P90/max: "
        f"{percentile(sent_lens, 50)} / {percentile(sent_lens, 90)} / {sent_lens[-1] if sent_lens else 0}"
    ) # 句子长度分布百分位数
    print(f"Sents with ≥1 lexicon head: {n_sents_with_head:,} ({hit_rate:.1f}%)")
    print(f"Unique heads observed: {len(head_hit_counts):,}")

    # ----Markdown 报告----
    lines = [
        "# Silver Corpus EDA (Step 1)",
        "",
        "> 原料是古典文献正文，**尚未**标注通假 span。",
        "> 本报告回答：有多少文本、句长是否适合 max_len=128、规范表能命中多少句子。",
        "",
        "## 1. 篇章规模",
        "",
        f"- Documents: **{n_docs:,}**",
        f"- Total characters: **{total_chars:,}**",
        f"- Lexicon tongjia heads: **{len(heads):,}**",
        "",
        "### By category",
        "",
        "| category | docs |",
        "|----------|-----:|",
    ]
    for cat, cnt in by_category.most_common():
        lines.append(f"| {cat} | {cnt:,} |")

    lines.extend(
        [
            "",
            "### Top sources",
            "",
            "| source | docs |",
            "|--------|-----:|",
        ]
    )
    for src, cnt in by_source.most_common(20):
        lines.append(f"| {src} | {cnt:,} |")

    n_le_128 = sum(1 for x in sent_lens if x <= 128)
    lines.extend(
        [
            "",
            "## 2. 句长分布（切句后）",
            "",
            f"- Sentences scored: **{n_sents:,}** "
            f"(cap {MAX_SENTS_PER_DOC} sents/doc)",
            f"- P50: **{percentile(sent_lens, 50)}**",
            f"- P90: **{percentile(sent_lens, 90)}**",
            f"- max: **{sent_lens[-1] if sent_lens else 0}**",
            f"- ≤128 chars: **{n_le_128:,}** "
            f"({100 * n_le_128 / max(n_sents, 1):.1f}%)",
            "",
            "## 3. 规范表覆盖（远监督信号强度）",
            "",
            f"- Sentences with ≥1 tongjia head: **{n_sents_with_head:,}** "
            f"({hit_rate:.1f}%)",
            f"- Unique heads seen in corpus: **{len(head_hit_counts):,}** / {len(heads):,}",
            "",
            "### Top-20 heads by sentence hits",
            "",
        ]
    )
    for ch, cnt in head_hit_counts.most_common(20):
        lines.append(f"- `{ch}`: {cnt:,}")


    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
