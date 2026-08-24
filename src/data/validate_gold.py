#!/usr/bin/env python3
"""校验已加工金标是否符合 ACPCR 规范草案。

检查内容：
  - 必填字段与枚举值（source/status/split）
  - span 边界、重叠、文本对齐（复用 gold_utils.validate_record）
  - ID 唯一、同一句子不跨 split（防泄露）
  - 正负比例、划分比例是否大致合理
  - 抽样 BIO 转换烟雾测试

用法：
  python src/data/validate_gold.py
  python src/data/validate_gold.py --path data/processed/gold/gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from gold_utils import GOLD_OUT, spans_to_bio, validate_record

# Windows 控制台对 Unicode 符号兼容性差，用 ASCII 标记
PASS = "[OK]"
FAIL = "[ERR]"
WARN = "[WARN]"
RESET = ""


def load_jsonl(path: Path) -> list[dict]:
    """读取 JSONL；任一行解析失败则直接退出。"""
    records = []
    with path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"{FAIL} {path.name}:{ln} JSON error: {exc}{RESET}")
                sys.exit(1)
    return records


def check_required(rec: dict) -> list[str]:
    """检查规范草案要求的字段与合法枚举。"""
    required = {"id", "text", "spans", "source", "status", "split"}
    missing = required - rec.keys()
    if missing:
        return [f"missing fields: {missing}"]
    if rec["source"] != "gold":
        return [f"source must be gold, got {rec['source']}"]
    if rec["status"] not in {"accepted", "uncertain", "rejected"}:
        return [f"invalid status: {rec['status']}"]
    if rec["split"] not in {"train", "dev", "test"}:
        return [f"invalid split: {rec['split']}"]
    return []


def check_no_leakage(records: list[dict]) -> list[str]:
    """同一原文不得同时出现在多个 split（防 train/test 泄露）。"""
    issues: list[str] = []
    by_text: dict[str, set[str]] = {}
    for rec in records:
        splits = by_text.setdefault(rec["text"], set())
        splits.add(rec["split"])
    for text, splits in by_text.items():
        if len(splits) > 1:
            issues.append(f"text in multiple splits: '{text[:20]}...' -> {splits}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=GOLD_OUT / "gold.jsonl",
        help="Gold JSONL to validate",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"{FAIL} not found: {args.path}{RESET}")
        sys.exit(1)

    records = load_jsonl(args.path)
    errors: list[str] = []
    warnings: list[str] = []

    # ---- 逐条：必填字段 + span 结构 ----
    ids = Counter()
    for rec in records:
        ids[rec.get("id", "")] += 1
        errors.extend(f"{rec.get('id')}: {e}" for e in check_required(rec))
        errors.extend(f"{rec.get('id')}: {e}" for e in validate_record(rec))

    dup_ids = [k for k, v in ids.items() if v > 1]
    if dup_ids:
        errors.append(f"duplicate ids: {dup_ids[:5]}")

    errors.extend(check_no_leakage(records))

    # ---- 分布告警（不阻断，仅提示）----
    # 任务书目标正例约 45–50%；这里放宽到 35–65% 作软约束
    pos_ratio = sum(1 for r in records if r["spans"]) / max(len(records), 1)
    if pos_ratio < 0.35 or pos_ratio > 0.65:
        warnings.append(f"positive ratio {pos_ratio:.2%} outside 35–65% band")

    split_counts = Counter(r["split"] for r in records)
    total = len(records)
    for split, expected in zip(("train", "dev", "test"), (0.8, 0.1, 0.1)):
        actual = split_counts.get(split, 0) / max(total, 1)
        if abs(actual - expected) > 0.05:
            warnings.append(f"{split} ratio {actual:.2%} deviates from {expected:.0%}")

    # 抽前 50 条做 BIO 烟雾测试
    for rec in records[:50]:
        try:
            spans_to_bio(rec["text"], rec["spans"])
        except ValueError as exc:
            errors.append(f"{rec['id']}: BIO failed — {exc}")

    print(f"Validated {len(records):,} records from {args.path}")
    if errors:
        print(f"{FAIL} {len(errors)} errors{RESET}")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)

    print(f"{PASS} structural validation passed{RESET}")
    if warnings:
        print(f"{WARN} {len(warnings)} warnings{RESET}")
        for w in warnings:
            print(f"  {w}")
    else:
        print(f"{PASS} no warnings{RESET}")


if __name__ == "__main__":
    main()
