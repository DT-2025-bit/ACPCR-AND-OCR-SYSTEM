#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""辅助指标：字级 F1、Top-K 高频字对 span-F1（报告用，主成绩仍可报全量 span-F1）。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from gold_utils import GOLD_OUT, iter_jsonl
from lexicon_filter import tags_to_spans


def pair_type(tongjia: str, benzi: str) -> str:
    return f"{tongjia}→{benzi}"


def type_from_span(span: dict) -> str:
    tj = (span.get("tongjia") or "").strip()
    bz = (span.get("benzi") or "").strip()
    if (not tj or not bz) and span.get("pair_id"):
        parts = str(span["pair_id"]).split("_")
        if len(parts) >= 3:
            return pair_type(parts[1], parts[2])
    return pair_type(tj, bz)


def load_top_pair_types(split: str = "train", k: int = 50) -> list[str]:
    """按金标 train 字对频次取 Top-K（键为 通假→本字）。"""
    ctr: Counter[str] = Counter()
    for rec in iter_jsonl(GOLD_OUT / f"{split}.jsonl"):
        for sp in rec.get("spans") or []:
            t = type_from_span(sp)
            if tj_ok(t):
                ctr[t] += 1
    return [t for t, _ in ctr.most_common(k)]


def tj_ok(t: str) -> bool:
    return bool(t) and "→" in t and not t.startswith("→") and not t.endswith("→")


def save_top_pair_types(types: list[str], path: Path | None = None) -> Path:
    path = path or (GOLD_OUT / "top50_pair_types.txt")
    path.write_text("\n".join(types) + "\n", encoding="utf-8")
    return path


def char_level_scores(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict[str, float]:
    """字级二分类：非 O = 通假字，严格逐字。"""
    tp = fp = fn = 0
    for gt, pr in zip(y_true, y_pred):
        m = min(len(gt), len(pr))
        for g, p in zip(gt[:m], pr[:m]):
            g_pos = g != "O"
            p_pos = p != "O"
            if p_pos and g_pos:
                tp += 1
            elif p_pos and not g_pos:
                fp += 1
            elif (not p_pos) and g_pos:
                fn += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return {
        "char_precision": float(p),
        "char_recall": float(r),
        "char_f1": float(f1),
        "char_tp": float(tp),
        "char_fp": float(fp),
        "char_fn": float(fn),
    }


def boundary_span_f1(
    gold_sets: Iterable[set[tuple[int, int]]],
    pred_sets: Iterable[set[tuple[int, int]]],
) -> dict[str, float]:
    tp = fp = fn = 0
    for gset, pset in zip(gold_sets, pred_sets):
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def topk_pair_scores(
    gold_span_lists: list[list[dict]],
    pred_tags_list: list[list[str]],
    top_types: list[str],
) -> dict[str, float]:
    """仅在「含 Top-K 金标 span」的句子上，对 Top-K 字对评边界 F1。

    口径（任务书辅助指标）：
    - 金标：type∈Top-K 的 (start,end)
    - 预测：该句全部预测 span
    - 不含 Top-K 金标的句子不计入（避免负例句上的误报淹没高频类）
    """
    allowed = set(top_types)
    gold_sets: list[set[tuple[int, int]]] = []
    pred_sets: list[set[tuple[int, int]]] = []
    n_sents = 0

    for gs, tags in zip(gold_span_lists, pred_tags_list):
        gset: set[tuple[int, int]] = set()
        for sp in gs:
            if type_from_span(sp) in allowed:
                gset.add((int(sp["start"]), int(sp["end"])))
        if not gset:
            continue
        n_sents += 1
        gold_sets.append(gset)
        pred_sets.append(set(tags_to_spans(tags)))

    scores = boundary_span_f1(gold_sets, pred_sets)
    n_gold = sum(len(s) for s in gold_sets)
    return {
        "top50_precision": scores["precision"],
        "top50_recall": scores["recall"],
        "top50_f1": scores["f1"],
        "top50_tp": scores["tp"],
        "top50_fp": scores["fp"],
        "top50_fn": scores["fn"],
        "top50_n_types": float(len(top_types)),
        "top50_gold_spans": float(n_gold),
        "top50_n_sents": float(n_sents),
    }
