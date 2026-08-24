#!/usr/bin/env python3
"""把通假字资源库原料转换成 ACPCR 金标 JSONL。

原料 → 加工产物：
  corpus/corpus.jsonl                 → 正例（专家标注位置 + 本字）
  evaluation/*_detection.jsonl        → 负例（检测任务中 output=[] 的句子）
  knowledge_base/tongjia_links.jsonl  → 规范表 lexicon.jsonl

输出目录 data/processed/gold/：
  lexicon.jsonl, gold.jsonl, train/dev/test.jsonl, bio/*.txt, stats.md

用法：
  python src/data/process_gold.py
  python src/data/process_gold.py --seed 42 --no-eval
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from gold_utils import (
    CORPUS_PATH,
    EVAL_DIR,
    GOLD_OUT,
    LEXICON_PATH,
    iter_jsonl,
    pair_id,
    parse_positions,
    strip_head_suffix,
    validate_record,
    write_jsonl,
)

# 规范要求的 train : dev : test ≈ 8 : 1 : 1
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def load_lexicon() -> dict[tuple[str, str], dict]:
    """加载规范表，键为 (通假字, 正字)。"""
    table: dict[tuple[str, str], dict] = {}
    for row in iter_jsonl(LEXICON_PATH):
        tj = strip_head_suffix(row["通假字"])
        bz = strip_head_suffix(row["正字"])
        key = (tj, bz)
        table[key] = {
            "pair_id": pair_id(tj, bz, row["通假字关系ID"]),
            "tongjia": tj,
            "benzi": bz,
            "pinyin": row.get("拼音", ""),
            "meaning": row.get("释义", ""),
            "relation_id": row["通假字关系ID"],
        }
    return table


def corpus_to_records(lexicon: dict[tuple[str, str], dict]) -> tuple[list[dict], list[str]]:
    """语料库 → 金标正例。

    原料一行 = 一句 + 一个通假字对 + 若干位置。
    同一句子若出现多次（不同字对），按 text 合并成一条，spans 并列。
    """
    grouped: dict[str, dict] = {}  # text → 中间桶
    warnings: list[str] = []

    for row in iter_jsonl(CORPUS_PATH):
        text = row["语料文本"].strip()
        tongjia = strip_head_suffix(row["通假字字头"])
        benzi = strip_head_suffix(row["正字字头"])
        positions = parse_positions(row["标注位置"])
        corpus_id = row["语料ID"]

        meta = {
            "corpus_id": corpus_id,
            "source_ref": row.get("出处", ""),
            "era": row.get("时代", ""),
            "meaning": row.get("释义", ""),
        }

        # 能命中规范表则复用其 pair_id；本字非空视为 high 置信
        lex = lexicon.get((tongjia, benzi))
        pid = lex["pair_id"] if lex else pair_id(tongjia, benzi)
        confidence = "high" if benzi else "low"

        new_spans = []
        for pos in positions:
            if pos < 0 or pos >= len(text):
                warnings.append(f"corpus#{corpus_id}: pos {pos} OOB in '{text[:20]}...'")
                continue
            span_len = max(len(tongjia), 1)
            chunk = text[pos : pos + span_len]
            # 词典字头与句中字形偶有异体差异（如 弈/奕）：以句中表面字形为准
            surface = chunk if chunk else tongjia
            if surface != tongjia:
                warnings.append(
                    f"corpus#{corpus_id}: pos {pos} surface '{surface}' != head '{tongjia}'"
                )
            new_spans.append(
                {
                    "start": pos,
                    "end": pos + len(surface),  # 右开区间
                    "type": "tongjia",
                    "tongjia": surface,
                    "benzi": benzi,
                    "confidence": confidence,
                    "pair_id": pid,
                    "head": tongjia,  # 临时字段，合并后删除
                }
            )

        if not new_spans:
            continue

        # 按整句聚合，避免同一句多条重复样本
        bucket = grouped.setdefault(
            text,
            {
                "text": text,
                "spans": [],
                "meta": [],
                "source": "gold",
                "status": "accepted",
                "negative_type": None,
            },
        )
        bucket["meta"].append(meta)
        bucket["spans"].extend(new_spans)

    records: list[dict] = []
    for idx, (text, bucket) in enumerate(sorted(grouped.items(), key=lambda x: x[0]), start=1):
        # 同一字符区间只保留一个 span；优先保留已填本字的版本
        by_range: dict[tuple[int, int], dict] = {}
        for span in bucket["spans"]:
            key = (span["start"], span["end"])
            prev = by_range.get(key)
            if prev is None:
                by_range[key] = span
                continue
            if not prev.get("benzi") and span.get("benzi"):
                by_range[key] = span
        spans = sorted(by_range.values(), key=lambda s: s["start"])
        for span in spans:
            span.pop("head", None)

        rec = {
            "id": f"gold_corpus_{idx:05d}",
            "text": text,
            "spans": spans,
            "source": "gold",
            "status": "accepted",
            "negative_type": None,
            "provenance": "tongjiazi_corpus",
            "corpus_ids": sorted({m["corpus_id"] for m in bucket["meta"]}),
        }
        # 入库前过一遍结构校验；不过则记 warning 并丢弃
        errs = validate_record(rec)
        if errs:
            warnings.append(f"{rec['id']}: rejected — {'; '.join(errs)}")
            continue
        records.append(rec)

    return records, warnings


def eval_detection_to_records(eval_files: list[Path]) -> tuple[list[dict], list[str]]:
    """从检测评测集导入样本。

    评测格式：input 里嵌套 JSON 字符串 {"sentence": "..."}，
    output 为通假位置列表；[] 表示负例（句中无通假）。
    """
    grouped: dict[str, dict] = {}
    warnings: list[str] = []
    counter = 0

    for path in eval_files:
        tag = path.stem
        for row in iter_jsonl(path):
            payload = json.loads(row["input"])  # 二次 JSON 解码
            text = payload["sentence"].strip()
            positions = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]
            if not isinstance(positions, list):
                warnings.append(f"{tag}: bad output for '{text[:16]}'")
                continue

            bucket = grouped.get(text)
            if bucket is None:
                counter += 1
                bucket = {
                    "id": f"gold_eval_{counter:05d}",
                    "text": text,
                    "spans": [],
                    "source": "gold",
                    "status": "accepted",
                    "negative_type": "normal" if not positions else None,
                    "provenance": f"tongjiazi_eval/{tag}",
                }
                grouped[text] = bucket
            elif positions and not bucket["spans"]:
                # 先前当负例、后来又出现正例标注 → 升为正例
                bucket["negative_type"] = None
            elif not positions and bucket["spans"]:
                # 正例优先，忽略冲突的负例标注
                warnings.append(f"{tag}: conflict pos/neg for '{text[:16]}' — keep positive")

            # 评测集通常只给位置、不给本字 → benzi 留空，confidence=low
            for pos in positions:
                if not isinstance(pos, int):
                    continue
                if pos < 0 or pos >= len(text):
                    warnings.append(f"{tag}: pos {pos} OOB")
                    continue
                char = text[pos]
                bucket["spans"].append(
                    {
                        "start": pos,
                        "end": pos + 1,
                        "type": "tongjia",
                        "tongjia": char,
                        "benzi": "",
                        "confidence": "low",
                        "pair_id": pair_id(char, "UNK"),
                    }
                )

    records: list[dict] = []
    for rec in grouped.values():
        uniq: dict[tuple, dict] = {}
        for span in rec["spans"]:
            key = (span["start"], span["end"])
            uniq[key] = span
        rec["spans"] = sorted(uniq.values(), key=lambda s: s["start"])
        if not rec["spans"]:
            rec["negative_type"] = rec.get("negative_type") or "normal"
        errs = validate_record(rec)
        if errs:
            warnings.append(f"{rec['id']}: rejected — {'; '.join(errs)}")
            continue
        records.append(rec)

    return records, warnings


def dedupe_corpus_eval(corpus: list[dict], eval_recs: list[dict]) -> list[dict]:
    """去掉评测集中已出现在语料正例里的句子，防止 train/test 文本撞车。"""
    corpus_texts = {r["text"] for r in corpus if r["spans"]}
    out = []
    for rec in eval_recs:
        if rec["text"] in corpus_texts:
            continue
        out.append(rec)
    return out


def stratify_key(rec: dict) -> str:
    """分层键：负例统一 NEG；正例用首个 span 的 pair_id。"""
    if not rec["spans"]:
        return "NEG"
    return rec["spans"][0].get("pair_id") or rec["spans"][0].get("tongjia", "UNK")


def assign_splits(
    records: list[dict],
    seed: int,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> list[dict]:
    """按 8:1:1 划分，正负例分开切，避免负例比例被打乱。

    正例先按字对桶打乱再拼接，减轻「某高频字对全进 train」的风险。
    """
    rng = random.Random(seed)

    positives = [r for r in records if r["spans"]]
    negatives = [r for r in records if not r["spans"]]

    def split_pool(pool: list[dict], _label: str) -> None:
        """对一个池子原地打乱后按比例写 split 字段。"""
        rng.shuffle(pool)
        n = len(pool)
        train_end = int(n * ratios[0])
        dev_end = train_end + int(n * ratios[1])
        for i, rec in enumerate(pool):
            if i < train_end:
                rec["split"] = "train"
            elif i < dev_end:
                rec["split"] = "dev"
            else:
                rec["split"] = "test"

    # 正例：先按字对分桶打乱，再拼成列表后整体 8:1:1
    pos_buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in positives:
        pos_buckets[stratify_key(rec)].append(rec)
    stratified_pos: list[dict] = []
    for items in pos_buckets.values():
        rng.shuffle(items)
        stratified_pos.extend(items)
    split_pool(stratified_pos, "pos")

    split_pool(negatives, "neg")

    result = stratified_pos + negatives
    rng.shuffle(result)
    return result


def export_bio(records: list[dict], path: Path) -> None:
    """导出逐字 BIO 文本，空行分隔句子，供序列标注训练直接读取。"""
    from gold_utils import spans_to_bio

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for rec in records:
        labels = spans_to_bio(rec["text"], rec["spans"])
        for ch, lab in zip(rec["text"], labels):
            lines.append(f"{ch}\t{lab}")
        lines.append("")  # 句间空行
    path.write_text("\n".join(lines), encoding="utf-8")


def build_stats(records: list[dict], warnings: list[str]) -> str:
    """生成 stats.md：规模、正负比、划分、高频字对、警告摘要。"""
    split_counts = Counter(r["split"] for r in records)
    pos = sum(1 for r in records if r["spans"])
    neg = len(records) - pos
    pair_counts = Counter(stratify_key(r) for r in records if r["spans"])
    benzi_filled = sum(1 for r in records for s in r["spans"] if s.get("benzi"))
    span_total = sum(len(r["spans"]) for r in records)

    lines = [
        "# Gold Label Stats",
        "",
        f"- Total records: **{len(records):,}**",
        f"- Positive (≥1 span): **{pos:,}** ({100 * pos / len(records):.1f}%)",
        f"- Negative (no span): **{neg:,}** ({100 * neg / len(records):.1f}%)",
        f"- Total spans: **{span_total:,}**",
        f"- Spans with benzi: **{benzi_filled:,}** ({100 * benzi_filled / max(span_total, 1):.1f}%)",
        f"- Unique pair types (positives): **{len(pair_counts):,}**",
        "",
        "## Split",
        "",
        "| split | count |",
        "|-------|------:|",
    ]
    for split in ("train", "dev", "test"):
        lines.append(f"| {split} | {split_counts.get(split, 0):,} |")

    lines.extend(["", "## Top-15 pair types", ""])
    for pid, cnt in pair_counts.most_common(15):
        lines.append(f"- `{pid}`: {cnt}")

    if warnings:
        lines.extend(["", f"## Warnings ({len(warnings)})", ""])
        for w in warnings[:30]:
            lines.append(f"- {w}")
        if len(warnings) > 30:
            lines.append(f"- ... and {len(warnings) - 30} more")

    return "\n".join(lines) + "\n"


def main() -> None:
    """流水线入口：规范表 → 正例 → 负例 → 划分 → 写出。"""
    parser = argparse.ArgumentParser(description="Process tongjiazi gold labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-eval-negatives", action="store_true", default=True)
    parser.add_argument("--no-eval", action="store_true", help="Skip eval import")
    args = parser.parse_args()

    print("Loading lexicon...", flush=True)
    lexicon = load_lexicon()
    lex_rows = list(lexicon.values())
    write_jsonl(GOLD_OUT / "lexicon.jsonl", lex_rows)
    print(f"  lexicon entries: {len(lex_rows)}")

    print("Converting corpus...", flush=True)
    corpus_records, w1 = corpus_to_records(lexicon)
    print(f"  corpus records: {len(corpus_records)}")

    all_warnings = list(w1)
    eval_records: list[dict] = []
    if not args.no_eval:
        eval_files = sorted(EVAL_DIR.glob("*_detection.jsonl"))
        print(f"Importing eval negatives from {len(eval_files)} files...", flush=True)
        eval_records, w2 = eval_detection_to_records(eval_files)
        # 当前策略：评测集只取负例，正例一律用专家语料（本字更全）
        eval_records = [r for r in eval_records if not r["spans"]]
        eval_records = dedupe_corpus_eval(corpus_records, eval_records)
        all_warnings.extend(w2)
        print(f"  eval negative records: {len(eval_records)}")

    combined = corpus_records + eval_records
    print(f"Assigning splits (seed={args.seed})...", flush=True)
    combined = assign_splits(combined, seed=args.seed)

    write_jsonl(GOLD_OUT / "gold.jsonl", combined)
    for split in ("train", "dev", "test"):
        subset = [r for r in combined if r["split"] == split]
        write_jsonl(GOLD_OUT / f"{split}.jsonl", subset)
        export_bio(subset, GOLD_OUT / "bio" / f"{split}.txt")

    stats = build_stats(combined, all_warnings)
    (GOLD_OUT / "stats.md").write_text(stats, encoding="utf-8")

    print(f"\nWrote {GOLD_OUT.relative_to(GOLD_OUT.parents[2])}/")
    print(stats)


if __name__ == "__main__":
    main()
