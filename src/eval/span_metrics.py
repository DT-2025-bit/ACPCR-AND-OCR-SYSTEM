#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位 span-F1 + 字级 F1 + Top-50 字对 F1 + 本字准确率。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from seqeval.metrics import f1_score, precision_score, recall_score

_SRC = Path(__file__).resolve().parents[1]
for p in (_SRC, _SRC / "data", _SRC / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import ID2LABEL
from extra_metrics import (
    char_level_scores,
    load_top_pair_types,
    save_top_pair_types,
    topk_pair_scores,
)
from lexicon_filter import constrain_benzi_by_lexicon, filter_pred_ids_by_lexicon

IGNORE_LABEL = -100

_TOP50_CACHE: list[str] | None = None


def get_top50_types() -> list[str]:
    global _TOP50_CACHE
    if _TOP50_CACHE is None:
        _TOP50_CACHE = load_top_pair_types("train", k=50)
        save_top_pair_types(_TOP50_CACHE)
    return _TOP50_CACHE


def ids_to_tags(label_ids: list[int]) -> list[str]:
    return [ID2LABEL[i] for i in label_ids if i != IGNORE_LABEL]


def gold_tags_from_batch(label_ids: torch.Tensor) -> list[list[str]]:
    return [ids_to_tags(row) for row in label_ids.tolist()]


def _valid_char_benzi(row_benzi: list[int], n_chars: int) -> list[int]:
    vals = [x for x in row_benzi if x != IGNORE_LABEL]
    return vals[:n_chars]


def _char_aligned_logits(
    logits_row: torch.Tensor,
    label_row: torch.Tensor,
    n_chars: int,
) -> torch.Tensor:
    keep = label_row != IGNORE_LABEL
    return logits_row[keep][:n_chars]


@torch.no_grad()
def eval_span_scores(
    model,
    loader,
    device: torch.device,
    max_batches: int = 0,
    lexicon_heads: set[str] | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    gold_span_lists: list[list[dict]] = []

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_ids = batch["label_ids"]
        chars_batch = batch.get("chars")
        gold_spans_batch = batch.get("gold_spans") or [[] for _ in range(len(label_ids))]

        preds = model.decode(input_ids, attention_mask, label_ids.to(device))
        golds = gold_tags_from_batch(label_ids)

        for bi, (g, p) in enumerate(zip(golds, preds)):
            if lexicon_heads is not None:
                chars = chars_batch[bi][: len(g)]
                p = filter_pred_ids_by_lexicon(p, chars, lexicon_heads)
            if len(p) != len(g):
                m = min(len(p), len(g))
                g, p = g[:m], p[:m]
            y_true.append(g)
            y_pred.append([ID2LABEL[tid] for tid in p])
            gold_span_lists.append(gold_spans_batch[bi])

        if max_batches > 0 and (i + 1) >= max_batches:
            break

    if was_training:
        model.train()

    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_sents": 0}

    out = {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "n_sents": float(len(y_true)),
    }
    out.update(char_level_scores(y_true, y_pred))
    out.update(topk_pair_scores(gold_span_lists, y_pred, get_top50_types()))
    return out


@torch.no_grad()
def eval_joint_scores(
    model,
    loader,
    device: torch.device,
    max_batches: int = 0,
    lexicon_heads: set[str] | None = None,
    tj_to_benzi: dict[str, list[str]] | None = None,
    benzi_stoi: dict[str, int] | None = None,
    benzi_itos: list[str] | None = None,
    top_pair_types: list[str] | None = None,
) -> dict[str, float]:
    """span-F1 + 字级 F1 + Top-50 字对 F1 + 本字准确率。"""
    was_training = model.training
    model.eval()

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    gold_span_lists: list[list[dict]] = []
    oracle_ok = oracle_n = 0
    pred_ok = pred_n = 0
    use_benzi_lex = tj_to_benzi is not None and benzi_stoi is not None
    if top_pair_types is None:
        top_pair_types = get_top50_types()

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_ids = batch["label_ids"]
        gold_benzi = batch["benzi_ids"]
        chars_batch = batch["chars"]
        gold_spans_batch = batch.get("gold_spans") or [[] for _ in chars_batch]

        bio_preds = model.decode(input_ids, attention_mask, label_ids.to(device))
        golds = gold_tags_from_batch(label_ids)

        if use_benzi_lex:
            logits = model.predict_benzi_logits(input_ids, attention_mask)
        else:
            benzi_pred = model.predict_benzi_ids(
                input_ids, attention_mask, label_ids.to(device)
            )

        for bi, (g_tags, p_ids) in enumerate(zip(golds, bio_preds)):
            chars = chars_batch[bi][: len(g_tags)]
            if lexicon_heads is not None:
                p_ids = filter_pred_ids_by_lexicon(p_ids, chars, lexicon_heads)
            if len(p_ids) != len(g_tags):
                m = min(len(p_ids), len(g_tags))
                g_tags, p_ids = g_tags[:m], p_ids[:m]

            pred_tags = [ID2LABEL[t] for t in p_ids]
            y_true.append(g_tags)
            y_pred.append(pred_tags)
            gold_span_lists.append(gold_spans_batch[bi])

            g_lab = [x for x in label_ids[bi].tolist() if x != IGNORE_LABEL][: len(g_tags)]
            g_bz = _valid_char_benzi(gold_benzi[bi].tolist(), len(g_tags))

            if use_benzi_lex:
                char_logits = _char_aligned_logits(
                    logits[bi], label_ids[bi].to(device), len(g_tags)
                )
                p_bz = constrain_benzi_by_lexicon(
                    char_logits, chars, tj_to_benzi, benzi_stoi
                )
            else:
                p_bz = _valid_char_benzi(benzi_pred[bi].tolist(), len(g_tags))

            m = min(len(g_lab), len(g_bz), len(p_bz), len(p_ids))
            g_lab, g_bz, p_bz, p_ids = g_lab[:m], g_bz[:m], p_bz[:m], p_ids[:m]

            for gl, gb, pb, pl in zip(g_lab, g_bz, p_bz, p_ids):
                if gl in (1, 2):
                    oracle_n += 1
                    if pb == gb:
                        oracle_ok += 1
                if pl in (1, 2) and gl in (1, 2):
                    pred_n += 1
                    if pb == gb:
                        pred_ok += 1

        if max_batches > 0 and (i + 1) >= max_batches:
            break

    if was_training:
        model.train()

    out: dict[str, float] = {
        "precision": float(precision_score(y_true, y_pred)) if y_true else 0.0,
        "recall": float(recall_score(y_true, y_pred)) if y_true else 0.0,
        "f1": float(f1_score(y_true, y_pred)) if y_true else 0.0,
        "n_sents": float(len(y_true)),
        "benzi_acc_oracle": oracle_ok / max(oracle_n, 1),
        "benzi_oracle_n": float(oracle_n),
        "benzi_acc_on_pred_tj": pred_ok / max(pred_n, 1),
        "benzi_pred_tj_n": float(pred_n),
        "benzi_lexicon_constrain": float(1.0 if use_benzi_lex else 0.0),
    }
    if y_true:
        out.update(char_level_scores(y_true, y_pred))
        out.update(topk_pair_scores(gold_span_lists, y_pred, top_pair_types))
    return out
