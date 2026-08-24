#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通假字识别 Demo（CLI）：输入句子 → 高亮位置 + 本字。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

_SRC = Path(__file__).resolve().parents[1]
_PROJECT = _SRC.parent
for p in (_SRC, _SRC / "data", _SRC / "model", _SRC / "eval", _SRC / "train"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benzi_vocab import BenziVocab, VOCAB_PATH
from dataset import ID2LABEL, LABEL2ID, BENZI_NONE
from lexicon_filter import (
    constrain_benzi_by_lexicon,
    filter_pred_ids_by_lexicon,
    load_tongjia_benzi_map,
    load_tongjia_heads,
    tags_to_spans,
)
from torch_dataset import encode_chars
from train_utils import load_checkpoint, resolve_model_name

IGNORE = -100


def predict_one(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    vocab: BenziVocab,
    heads: set[str] | None,
    tj_map: dict[str, list[str]],
) -> dict:
    chars = list(text)
    label_ids = [LABEL2ID["O"]] * len(chars)
    benzi_ids = [vocab.none_id] * len(chars)
    enc = encode_chars(chars, label_ids, benzi_ids, tokenizer)
    input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [enc["attention_mask"]], dtype=torch.long, device=device
    )
    lab = torch.tensor([enc["label_ids"]], dtype=torch.long, device=device)

    with torch.no_grad():
        # CRF decode 只返回 mask 有效位（与字级对齐，不含 CLS/SEP）
        pred_char_ids = model.decode(input_ids, attention_mask, lab)[0]
        logits = model.predict_benzi_logits(input_ids, attention_mask)[0]

    keep = [i for i, y in enumerate(enc["label_ids"]) if y != IGNORE]
    n = min(len(chars), len(pred_char_ids), len(keep))
    chars = chars[:n]
    pred_ids = list(pred_char_ids[:n])
    char_logits = logits[keep][:n]

    if heads is not None:
        pred_ids = filter_pred_ids_by_lexicon(pred_ids, chars, heads)
    tags = [ID2LABEL[i] for i in pred_ids]
    bz_ids = constrain_benzi_by_lexicon(char_logits, chars, tj_map, vocab.stoi)

    spans = []
    for start, end in tags_to_spans(tags):
        surface = "".join(chars[start:end])
        benzi = vocab.decode(bz_ids[start]) if start < len(bz_ids) else BENZI_NONE
        if benzi in (BENZI_NONE, "[UNK]"):
            benzi = ""
        spans.append(
            {
                "start": start,
                "end": end,
                "tongjia": surface,
                "benzi": benzi,
            }
        )

    # 高亮：通假【字/本字】
    pieces = []
    i = 0
    starts = {s["start"]: s for s in spans}
    while i < len(chars):
        if i in starts:
            s = starts[i]
            e = s["end"]
            if s["benzi"]:
                pieces.append(f"【{s['tongjia']}→{s['benzi']}】")
            else:
                pieces.append(f"【{s['tongjia']}】")
            i = e
        else:
            pieces.append(chars[i])
            i += 1

    return {
        "text": text,
        "spans": spans,
        "display": "".join(pieces),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="通假字定位 + 本字预测 Demo")
    parser.add_argument(
        "--ckpt",
        default=str(_PROJECT / "checkpoints" / "gold_full_v2" / "best.pt"),
    )
    parser.add_argument("--text", type=str, default="", help="单句推理")
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="每行一句；与 --text 二选一",
    )
    parser.add_argument("--no-lexicon-filter", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(Path(args.ckpt), device=device)
    model.eval()
    print("device:", device)
    print("ckpt:", args.ckpt)

    vocab_path = Path(args.ckpt).parent / "benzi_vocab.json"
    vocab = BenziVocab.load(vocab_path if vocab_path.exists() else VOCAB_PATH)
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(cfg["model_name"]))
    heads = None if args.no_lexicon_filter else load_tongjia_heads()
    tj_map = load_tongjia_benzi_map()

    texts: list[str] = []
    if args.text:
        texts = [args.text.strip()]
    elif args.file:
        texts = [
            ln.strip()
            for ln in Path(args.file).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    else:
        # 交互
        print("输入句子（空行退出）：")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            texts.append(line)
            if len(texts) >= 1 and not sys.stdin.isatty():
                break

    if not texts:
        # 默认演示句
        texts = [
            "塗金披繡，漿酒藿肉者，故不可稱紀。",
            "知之为知之，不知为不知，是知也。",
        ]
        print("（未提供输入，使用内置样例）")

    for text in texts:
        out = predict_one(
            model,
            tokenizer,
            text,
            device,
            vocab=vocab,
            heads=heads,
            tj_map=tj_map,
        )
        print("-" * 60)
        print("原文:", out["text"])
        print("高亮:", out["display"])
        if out["spans"]:
            for s in out["spans"]:
                bz = s["benzi"] or "（未预测）"
                print(f"  span [{s['start']},{s['end']}) {s['tongjia']} → {bz}")
        else:
            print("  （未检出通假）")


if __name__ == "__main__":
    main()
