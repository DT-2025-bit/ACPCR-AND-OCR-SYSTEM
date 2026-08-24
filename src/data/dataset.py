#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金标 → 训练样本（字级 BIO）。

一句话：把一条金标 JSON，变成「每个字一个标签」。
例如：
  文本：  舍 之
  标签：  B-TJ O
  本字：  釋 [NONE]
"""

from __future__ import annotations

from typing import Any

from gold_utils import GOLD_OUT, iter_jsonl, spans_to_bio

# 三种位置标签 → 数字
LABEL2ID = {"O": 0, "B-TJ": 1, "I-TJ": 2}
ID2LABEL = {0: "O", 1: "B-TJ", 2: "I-TJ"}

# 这个位置不是通假时，本字栏填这个占位符
BENZI_NONE = "[NONE]"


def record_to_example(rec: dict[str, Any]) -> dict[str, Any]:
    """把金标里的一条记录，转成训练用的一条样本。"""
    text = rec["text"]  # 整句字符串，如 "高祖舍之"

    spans = rec.get("spans") or []

    # spans_to_bio：根据 spans，给每个字打上 O / B-TJ / I-TJ
    bio = spans_to_bio(text, spans)  # 例如 ["O","O","B-TJ","O"]
    label_ids = [LABEL2ID[tag] for tag in bio]  # 转成 [0,0,1,0]

    # 本字：默认全是 [NONE]；只有通假那段填正字
    benzi = [BENZI_NONE] * len(text)
    for sp in spans:
        b = sp.get("benzi") or BENZI_NONE
        for i in range(sp["start"], sp["end"]):
            benzi[i] = b

    # 保留 span 供 Top-K 字对评测
    gold_spans = [
        {
            "start": int(sp["start"]),
            "end": int(sp["end"]),
            "tongjia": (sp.get("tongjia") or text[sp["start"] : sp["end"]]),
            "benzi": (sp.get("benzi") or ""),
            "pair_id": sp.get("pair_id") or "",
        }
        for sp in spans
    ]

    return {
        "id": rec["id"],
        "chars": list(text),
        "label_ids": label_ids,
        "benzi": benzi,
        "gold_spans": gold_spans,
        "source": rec.get("source", "gold"),
    }


def load_split(name: str) -> list[dict[str, Any]]:
    """name 填 train / dev / test，读取对应 jsonl 并全部转换。"""
    path = GOLD_OUT / f"{name}.jsonl"
    return [record_to_example(r) for r in iter_jsonl(path)]


def load_records(path) -> list[dict[str, Any]]:
    """任意 JSONL（金标/银标 warmup）→ 训练样本。"""
    from pathlib import Path

    return [record_to_example(r) for r in iter_jsonl(Path(path))]


if __name__ == "__main__":
    train = load_split("train")
    ex = train[0]

    print("id:", ex["id"])
    print("句子前40字:", "".join(ex["chars"])[:40])
    print("标签前40个:", [ID2LABEL[i] for i in ex["label_ids"][:40]])

    hits = [
        (c, b)
        for c, b, y in zip(ex["chars"], ex["benzi"], ex["label_ids"])
        if y != 0
    ]
    print("通假位置(字,本字):", hits[:10])
    print("训练集条数:", len(train))
