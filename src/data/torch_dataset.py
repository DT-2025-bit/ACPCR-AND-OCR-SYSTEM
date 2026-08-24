#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把金标样本变成 PyTorch DataLoader 能用的 batch（BertTokenizer + 本字）。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from benzi_vocab import BenziVocab, build_benzi_vocab_from_train
from dataset import load_records, load_split
from gold_utils import PROJECT_ROOT

IGNORE_LABEL = -100
MODEL_NAME = "models/chinese-bert-wwm-ext"
SILVER_WARMUP_PATH = (
    PROJECT_ROOT / "data" / "processed" / "silver" / "silver_warmup.jsonl"
)


def encode_chars(
    chars: list[str],
    label_ids: list[int],
    benzi_ids: list[int],
    tokenizer,
) -> dict:
    """汉字列表 + 字级标签/本字 → BERT 输入并对齐。"""
    enc = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=128,
        return_tensors=None,
    )
    word_ids = enc.word_ids()
    aligned_labels = []
    aligned_benzi = []
    for wid in word_ids:
        if wid is None:
            aligned_labels.append(IGNORE_LABEL)
            aligned_benzi.append(IGNORE_LABEL)
        else:
            aligned_labels.append(label_ids[wid])
            aligned_benzi.append(benzi_ids[wid])
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "label_ids": aligned_labels,
        "benzi_ids": aligned_benzi,
    }


class TongjiaDataset(Dataset):
    """包装 load_split / 任意 jsonl；把本字编成 id。"""

    def __init__(
        self,
        split: str = "train",
        benzi_vocab: BenziVocab | None = None,
        *,
        examples: list[dict] | None = None,
        jsonl_path=None,
        pos_oversample: int = 1,
    ):
        if examples is not None:
            self.examples = list(examples)
        elif jsonl_path is not None:
            self.examples = load_records(jsonl_path)
        else:
            self.examples = load_split(split)
        self.benzi_vocab = benzi_vocab or build_benzi_vocab_from_train()

        # 正例过采样：提升召回（报告主模型常用）
        k = max(int(pos_oversample), 1)
        if k > 1:
            pos = [
                ex
                for ex in self.examples
                if any(y != 0 for y in ex["label_ids"])
            ]
            self.examples = self.examples + pos * (k - 1)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        benzi_ids = [self.benzi_vocab.encode(b) for b in ex["benzi"]]
        return {
            **ex,
            "benzi_ids": benzi_ids,
        }


def collate_batch(batch: list[dict], tokenizer) -> dict:
    """多条样本补齐成一个 batch。"""
    encoded = [
        encode_chars(ex["chars"], ex["label_ids"], ex["benzi_ids"], tokenizer)
        for ex in batch
    ]
    max_len = max(len(x["input_ids"]) for x in encoded)

    input_ids, attention_mask, label_ids, benzi_ids = [], [], [], []
    for ex in encoded:
        n = len(ex["input_ids"])
        pad_n = max_len - n
        input_ids.append(ex["input_ids"] + [tokenizer.pad_token_id] * pad_n)
        attention_mask.append(ex["attention_mask"] + [0] * pad_n)
        label_ids.append(ex["label_ids"] + [IGNORE_LABEL] * pad_n)
        benzi_ids.append(ex["benzi_ids"] + [IGNORE_LABEL] * pad_n)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "label_ids": torch.tensor(label_ids, dtype=torch.long),
        "benzi_ids": torch.tensor(benzi_ids, dtype=torch.long),
        "ids": [ex["id"] for ex in batch],
        "chars": [ex["chars"] for ex in batch],
        "gold_spans": [ex.get("gold_spans") or [] for ex in batch],
        "gold_benzi_str": [ex["benzi"] for ex in batch],
    }


if __name__ == "__main__":
    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab = build_benzi_vocab_from_train()
    vocab.save()
    ds = TongjiaDataset("train", vocab)

    def _collate(batch):
        return collate_batch(batch, tokenizer)

    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=_collate)
    batch = next(iter(loader))
    print("benzi vocab:", len(vocab))
    print("benzi_ids shape:", batch["benzi_ids"].shape)
    print("label 集合:", sorted(batch["label_ids"].unique().tolist()))
