#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测：频次头部（累计覆盖 train 通假实例 ≥25%）及「类型数前 25%」。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]
_PROJECT = _SRC.parent
for p in (_SRC, _SRC / "data", _SRC / "model", _SRC / "eval", _SRC / "train"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benzi_vocab import BenziVocab
from extra_metrics import type_from_span
from gold_utils import GOLD_OUT, iter_jsonl
from lexicon_filter import load_tongjia_benzi_map, load_tongjia_heads
from span_metrics import eval_joint_scores
from torch_dataset import TongjiaDataset, collate_batch
from train_utils import load_checkpoint, make_tokenizer


def train_pair_counts() -> Counter[str]:
    ctr: Counter[str] = Counter()
    for rec in iter_jsonl(GOLD_OUT / "train.jsonl"):
        for sp in rec.get("spans") or []:
            t = type_from_span(sp)
            if t and "→" in t and not t.startswith("→") and not t.endswith("→"):
                ctr[t] += 1
    return ctr


def head_by_span_cover(ctr: Counter[str], cover: float = 0.25) -> tuple[list[str], float]:
    total = sum(ctr.values())
    ranked = ctr.most_common()
    types: list[str] = []
    cum = 0
    for t, v in ranked:
        types.append(t)
        cum += v
        if cum / total >= cover:
            return types, cum / total
    return types, cum / max(total, 1)


def head_by_type_quantile(ctr: Counter[str], q: float = 0.25) -> tuple[list[str], float]:
    ranked = ctr.most_common()
    k = max(1, int(round(len(ranked) * q)))
    types = [t for t, _ in ranked[:k]]
    total = sum(ctr.values())
    cov = sum(v for _, v in ranked[:k]) / max(total, 1)
    return types, cov


def main() -> None:
    ckpt = _PROJECT / "checkpoints" / "gold_full_v2" / "best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(ckpt, device=device)
    model.eval()

    vocab = BenziVocab.load(ckpt.parent / "benzi_vocab.json")
    tok = make_tokenizer(cfg)
    loader = DataLoader(
        TongjiaDataset("test", vocab),
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tok),
    )
    heads = load_tongjia_heads()
    tj_map = load_tongjia_benzi_map()

    ctr = train_pair_counts()
    types_span25, cov_span25 = head_by_span_cover(ctr, 0.25)
    types_q25, cov_q25 = head_by_type_quantile(ctr, 0.25)

    print(f"train unique types={len(ctr)}  spans={sum(ctr.values())}")
    print(
        f"[A] 累计覆盖≥25%实例: K={len(types_span25)}  "
        f"train_cover={cov_span25*100:.2f}%"
    )
    print(
        f"[B] 类型数前25%:       K={len(types_q25)}  "
        f"train_cover={cov_q25*100:.2f}%"
    )

    s_full = eval_joint_scores(
        model,
        loader,
        device,
        lexicon_heads=heads,
        tj_to_benzi=tj_map,
        benzi_stoi=vocab.stoi,
        top_pair_types=types_span25,
    )
    # reload loader iterator — need fresh loader
    loader = DataLoader(
        TongjiaDataset("test", vocab),
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tok),
    )
    s_q = eval_joint_scores(
        model,
        loader,
        device,
        lexicon_heads=heads,
        tj_to_benzi=tj_map,
        benzi_stoi=vocab.stoi,
        top_pair_types=types_q25,
    )

    print("\n=== 全量 span（对照）===")
    print(
        f"P/R/F1 = {s_full['precision']:.4f} / {s_full['recall']:.4f} / {s_full['f1']:.4f}"
    )
    print("\n=== [A] 前25%实例覆盖（高频头）===")
    print(
        f"P/R/F1 = {s_full['top50_precision']:.4f} / "
        f"{s_full['top50_recall']:.4f} / {s_full['top50_f1']:.4f}  "
        f"gold_spans={int(s_full['top50_gold_spans'])}  "
        f"sents={int(s_full['top50_n_sents'])}"
    )
    print("\n=== [B] 类型数前25% ===")
    print(
        f"P/R/F1 = {s_q['top50_precision']:.4f} / "
        f"{s_q['top50_recall']:.4f} / {s_q['top50_f1']:.4f}  "
        f"gold_spans={int(s_q['top50_gold_spans'])}  "
        f"sents={int(s_q['top50_n_sents'])}"
    )

    out = {
        "ckpt": str(ckpt),
        "full_span": {
            "precision": s_full["precision"],
            "recall": s_full["recall"],
            "f1": s_full["f1"],
        },
        "head_span_cover_25pct": {
            "K": len(types_span25),
            "train_span_cover": cov_span25,
            "precision": s_full["top50_precision"],
            "recall": s_full["top50_recall"],
            "f1": s_full["top50_f1"],
            "gold_spans": s_full["top50_gold_spans"],
            "n_sents": s_full["top50_n_sents"],
        },
        "head_type_quantile_25pct": {
            "K": len(types_q25),
            "train_span_cover": cov_q25,
            "precision": s_q["top50_precision"],
            "recall": s_q["top50_recall"],
            "f1": s_q["top50_f1"],
            "gold_spans": s_q["top50_gold_spans"],
            "n_sents": s_q["top50_n_sents"],
        },
    }
    path = ckpt.parent / "metrics_test_head25.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote:", path)


if __name__ == "__main__":
    main()
