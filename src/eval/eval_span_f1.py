#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金标 test/dev：span-F1 + 字级 F1 + Top-50 + 本字准确率。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]
_PROJECT = _SRC.parent
for p in (_SRC, _SRC / "data", _SRC / "model", _SRC / "eval", _SRC / "train"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benzi_vocab import BenziVocab, VOCAB_PATH
from lexicon_filter import load_tongjia_benzi_map, load_tongjia_heads
from span_metrics import eval_joint_scores
from torch_dataset import TongjiaDataset, collate_batch
from train_utils import load_checkpoint, make_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=str(_PROJECT / "checkpoints" / "gold_full_v2" / "best.pt"),
    )
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lexicon-filter",
        choices=["auto", "on", "off"],
        default="auto",
    )
    parser.add_argument(
        "--benzi-lexicon",
        choices=["auto", "on", "off"],
        default="auto",
        help="本字是否用规范表候选约束；auto=开",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        # 兼容旧主模型路径
        alt = _PROJECT / "checkpoints" / "gold_full" / "best.pt"
        if alt.exists():
            print(f"warn: {ckpt_path} 不存在，改用 {alt}")
            ckpt_path = alt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt = load_checkpoint(ckpt_path, device=device)
    model.eval()

    print("load:", ckpt_path)
    print(
        "ckpt epoch:",
        ckpt.get("epoch"),
        "dev_f1:",
        ckpt.get("dev_f1"),
        "use_lora:",
        cfg.get("use_lora", False),
    )

    if args.lexicon_filter == "auto":
        use_lex = bool(cfg.get("lexicon_filter", False))
    else:
        use_lex = args.lexicon_filter == "on"
    lexicon_heads = load_tongjia_heads() if use_lex else None
    print("lexicon_filter (span):", "on" if lexicon_heads else "off")

    use_benzi_lex = args.benzi_lexicon != "off"
    tj_to_benzi = load_tongjia_benzi_map() if use_benzi_lex else None
    print("benzi_lexicon_constrain:", "on" if use_benzi_lex else "off")

    vocab_path = ckpt_path.parent / "benzi_vocab.json"
    if not vocab_path.exists():
        vocab_path = VOCAB_PATH
    benzi_vocab = BenziVocab.load(vocab_path)
    print("benzi vocab:", len(benzi_vocab))

    tokenizer = make_tokenizer(cfg)

    def _collate(batch):
        return collate_batch(batch, tokenizer)

    loader = DataLoader(
        TongjiaDataset(args.split, benzi_vocab),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    scores = eval_joint_scores(
        model,
        loader,
        device,
        lexicon_heads=lexicon_heads,
        tj_to_benzi=tj_to_benzi,
        benzi_stoi=benzi_vocab.stoi if use_benzi_lex else None,
    )
    metrics = {
        "split": args.split,
        "ckpt": str(ckpt_path),
        "model_name": cfg.get("model_name"),
        "use_lora": bool(cfg.get("use_lora", False)),
        "lexicon_filter": bool(lexicon_heads),
        "benzi_lexicon_constrain": use_benzi_lex,
        **scores,
    }
    print(
        "span  P/R/F1 = "
        f"{scores['precision']:.4f} / {scores['recall']:.4f} / {scores['f1']:.4f}"
    )
    print(
        "char  P/R/F1 = "
        f"{scores.get('char_precision', 0):.4f} / "
        f"{scores.get('char_recall', 0):.4f} / "
        f"{scores.get('char_f1', 0):.4f}"
    )
    print(
        "top50 P/R/F1 = "
        f"{scores.get('top50_precision', 0):.4f} / "
        f"{scores.get('top50_recall', 0):.4f} / "
        f"{scores.get('top50_f1', 0):.4f}  "
        f"(gold_spans={int(scores.get('top50_gold_spans', 0))}, "
        f"sents={int(scores.get('top50_n_sents', 0))})"
    )
    print(
        "benzi_acc_oracle = "
        f"{scores['benzi_acc_oracle']:.4f}  n={int(scores['benzi_oracle_n'])}"
    )
    print(
        "benzi_acc_on_pred_tj = "
        f"{scores['benzi_acc_on_pred_tj']:.4f}  n={int(scores['benzi_pred_tj_n'])}"
    )
    print("summary:", json.dumps(metrics, ensure_ascii=False, indent=2))

    out = ckpt_path.parent / f"metrics_{args.split}.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote:", out)


if __name__ == "__main__":
    main()
