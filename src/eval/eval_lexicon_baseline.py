#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""词典匹配基线（消融 A0）：字头命中即标通假 + 规范表优先本字。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from seqeval.metrics import f1_score, precision_score, recall_score

_SRC = Path(__file__).resolve().parents[1]
_PROJECT = _SRC.parent
for p in (_SRC, _SRC / "data", _SRC / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import ID2LABEL, LABEL2ID, BENZI_NONE, load_split
from extra_metrics import (
    char_level_scores,
    load_top_pair_types,
    save_top_pair_types,
    topk_pair_scores,
)
from gold_utils import GOLD_OUT
from lexicon_filter import load_tongjia_benzi_map, load_tongjia_heads


def predict_tags(chars: list[str], heads: set[str]) -> list[str]:
    return ["B-TJ" if c in heads else "O" for c in chars]


def predict_benzi(
    chars: list[str],
    tags: list[str],
    tj_to_benzi: dict[str, list[str]],
) -> list[str]:
    from lexicon_filter import tags_to_spans

    out = [BENZI_NONE] * len(chars)
    for start, end in tags_to_spans(tags):
        surface = "".join(chars[start:end])
        cands = tj_to_benzi.get(surface) or []
        bz = cands[0] if cands else BENZI_NONE
        for i in range(start, end):
            out[i] = bz
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    heads = load_tongjia_heads()
    tj_to_benzi = load_tongjia_benzi_map()
    examples = load_split(args.split)
    top_types = load_top_pair_types("train", k=50)
    save_top_pair_types(top_types)
    print(f"split={args.split}  n={len(examples)}  heads={len(heads)}  top50={len(top_types)}")

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    gold_span_lists: list[list[dict]] = []
    oracle_ok = oracle_n = 0
    pred_ok = pred_n = 0

    for ex in examples:
        chars = ex["chars"]
        gold_tags = [ID2LABEL[i] for i in ex["label_ids"]]
        pred_tags = predict_tags(chars, heads)
        y_true.append(gold_tags)
        y_pred.append(pred_tags)
        gold_span_lists.append(ex.get("gold_spans") or [])

        gold_benzi = ex["benzi"]
        pred_benzi = predict_benzi(chars, pred_tags, tj_to_benzi)
        gold_ids = ex["label_ids"]
        pred_ids = [LABEL2ID[t] for t in pred_tags]

        for gl, gb, pb, pl in zip(gold_ids, gold_benzi, pred_benzi, pred_ids):
            if gl in (1, 2):
                oracle_n += 1
                if pb == gb:
                    oracle_ok += 1
            if pl in (1, 2) and gl in (1, 2):
                pred_n += 1
                if pb == gb:
                    pred_ok += 1

    char_scores = char_level_scores(y_true, y_pred)
    top50 = topk_pair_scores(gold_span_lists, y_pred, top_types)

    metrics = {
        "split": args.split,
        "method": "lexicon_match_heads",
        "n_sents": len(examples),
        "n_heads": len(heads),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "benzi_acc_oracle": oracle_ok / max(oracle_n, 1),
        "benzi_oracle_n": oracle_n,
        "benzi_acc_on_pred_tj": pred_ok / max(pred_n, 1),
        "benzi_pred_tj_n": pred_n,
        **char_scores,
        **top50,
    }

    print(
        "span  P/R/F1 = "
        f"{metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}"
    )
    print(
        "char  P/R/F1 = "
        f"{metrics['char_precision']:.4f} / {metrics['char_recall']:.4f} / {metrics['char_f1']:.4f}"
    )
    print(
        "top50 P/R/F1 = "
        f"{metrics['top50_precision']:.4f} / {metrics['top50_recall']:.4f} / {metrics['top50_f1']:.4f}"
    )

    out = Path(args.out) if args.out else (
        _PROJECT / "checkpoints" / "lexicon_baseline" / f"metrics_{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote:", out)

    mirror = GOLD_OUT / f"lexicon_baseline_{args.split}.json"
    mirror.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote:", mirror)


if __name__ == "__main__":
    main()
