#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通假规范表约束：定位字头过滤 + 本字候选约束。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from dataset import ID2LABEL, LABEL2ID
from gold_utils import GOLD_OUT, iter_jsonl


def load_tongjia_heads(lexicon_path: Path | None = None) -> set[str]:
    """加载规范表通假字头（当前均为单字）。"""
    path = lexicon_path or (GOLD_OUT / "lexicon.jsonl")
    heads: set[str] = set()
    for row in iter_jsonl(path):
        tj = (row.get("tongjia") or "").strip()
        if tj:
            heads.add(tj)
    return heads


def load_tongjia_benzi_map(lexicon_path: Path | None = None) -> dict[str, list[str]]:
    """通假字头 → 本字候选列表（去重、保序）。"""
    path = lexicon_path or (GOLD_OUT / "lexicon.jsonl")
    mapping: dict[str, list[str]] = defaultdict(list)
    for row in iter_jsonl(path):
        tj = (row.get("tongjia") or "").strip()
        bz = (row.get("benzi") or "").strip()
        if not tj or not bz:
            continue
        if bz not in mapping[tj]:
            mapping[tj].append(bz)
    return dict(mapping)


def tags_to_spans(tags: list[str]) -> list[tuple[int, int]]:
    """BIO 标签 → [start, end) span 列表。"""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(tags)
    while i < n:
        if tags[i] == "B-TJ":
            j = i + 1
            while j < n and tags[j] == "I-TJ":
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def filter_tags_by_lexicon(
    tags: list[str],
    chars: list[str],
    heads: set[str],
) -> list[str]:
    """只保留表面形式在规范表中的通假 span。"""
    if len(tags) != len(chars):
        m = min(len(tags), len(chars))
        tags, chars = tags[:m], chars[:m]
    out = ["O"] * len(tags)
    for start, end in tags_to_spans(tags):
        surface = "".join(chars[start:end])
        if surface in heads:
            out[start] = "B-TJ"
            for k in range(start + 1, end):
                out[k] = "I-TJ"
    return out


def filter_pred_ids_by_lexicon(
    pred_ids: list[int],
    chars: list[str],
    heads: set[str],
) -> list[int]:
    tags = [ID2LABEL[i] for i in pred_ids]
    filtered = filter_tags_by_lexicon(tags, chars, heads)
    return [LABEL2ID[t] for t in filtered]


def constrain_benzi_by_lexicon(
    logits: torch.Tensor,
    chars: list[str],
    tj_to_benzi: dict[str, list[str]],
    benzi_stoi: dict[str, int],
) -> list[int]:
    """按字面查规范表，只在候选本字里取 logits 最大者。

    logits: [L, V]，与 chars 对齐（不含 CLS/SEP/PAD）。
    """
    if logits.dim() != 2:
        raise ValueError("logits 应为 [L, V]")
    L = min(logits.size(0), len(chars))
    out: list[int] = []
    for i in range(L):
        ch = chars[i]
        cands = tj_to_benzi.get(ch) or []
        cand_ids = [benzi_stoi[b] for b in cands if b in benzi_stoi]
        if cand_ids:
            row = logits[i, cand_ids]
            out.append(int(cand_ids[int(row.argmax().item())]))
        else:
            out.append(int(logits[i].argmax().item()))
    return out
