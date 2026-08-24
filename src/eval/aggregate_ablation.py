#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 metrics → 面试友好消融表（含字级 / Top-50）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
CKPT = _PROJECT / "checkpoints"

ROWS = [
    ("A0", "词典匹配基线", "lexicon_baseline/metrics_test.json"),
    ("A4-full", "金标全参 (v1)", "gold_full/metrics_test.json"),
    ("主模型", "金标全参加强 (v2)", "gold_full_v2/metrics_test.json"),
    ("A4-lora", "金标 LoRA", "gold_lora/metrics_test.json"),
    ("A1-full", "金标全参 + 银标预热", "gold_full_silver_warmup/metrics_test.json"),
]


def load_metrics(rel: str) -> dict | None:
    path = CKPT / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(_PROJECT / "docs" / "消融结果表.md"),
    )
    args = parser.parse_args()

    lines = [
        "# 消融与主指标（自动汇总）",
        "",
        "> 主成绩：全量 **span-F1**。辅助：字级 F1、Top-50 高频字对 F1（任务书允许的分层指标）。",
        "",
        "| 编号 | 设定 | span-F1 | 字级 F1 | Top-50 F1 | 本字 Acc |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    missing = []
    for aid, name, rel in ROWS:
        m = load_metrics(rel)
        if m is None:
            missing.append(rel)
            lines.append(f"| {aid} | {name} | — | — | — | — |")
        else:
            lines.append(
                f"| {aid} | {name} | "
                f"{fmt(m.get('f1'))} | {fmt(m.get('char_f1'))} | "
                f"{fmt(m.get('top50_f1'))} | {fmt(m.get('benzi_acc_oracle'))} |"
            )

    lines.extend(
        [
            "",
            "## span 明细",
            "",
            "| 编号 | P | R | F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for aid, name, rel in ROWS:
        m = load_metrics(rel)
        if m is None:
            lines.append(f"| {aid} | — | — | — |")
        else:
            lines.append(
                f"| {aid} | {fmt(m.get('precision'))} | "
                f"{fmt(m.get('recall'))} | {fmt(m.get('f1'))} |"
            )

    lines.extend(["", "## 文件", ""])
    for aid, name, rel in ROWS:
        status = "✓" if (CKPT / rel).exists() else "✗ missing"
        lines.append(f"- `{aid}` {name}: `checkpoints/{rel}` ({status})")

    if missing:
        lines.extend(["", "## 待补", ""])
        for rel in missing:
            lines.append(f"- `checkpoints/{rel}`")

    lines.extend(
        [
            "",
            "## 面试口径（建议）",
            "",
            "- 对外主成绩：金标 test **span-F1**（严格边界）。",
            "- 业务侧可补充：字级 F1、Top-50 高频字对 F1（覆盖常见通假，更贴近阅读辅助场景）。",
            "- 相对词典基线的提升倍数，优先写进摘要。",
            "",
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote:", out)
    if missing:
        print(f"missing {len(missing)} metrics.")


if __name__ == "__main__":
    main()
