#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本字词表：从金标 train 统计，供分类头使用。"""

from __future__ import annotations

import json
from pathlib import Path

from dataset import BENZI_NONE, load_split
from gold_utils import GOLD_OUT

VOCAB_PATH = GOLD_OUT / "benzi_vocab.json"
UNK = "[UNK]"


class BenziVocab:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        if BENZI_NONE not in self.stoi:
            raise ValueError("vocab 必须包含 [NONE]")
        if UNK not in self.stoi:
            raise ValueError("vocab 必须包含 [UNK]")

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def none_id(self) -> int:
        return self.stoi[BENZI_NONE]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, benzi: str) -> int:
        return self.stoi.get(benzi, self.unk_id)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return UNK

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, obj: dict) -> "BenziVocab":
        return cls(obj["itos"])

    def save(self, path: Path | None = None) -> Path:
        path = path or VOCAB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "BenziVocab":
        path = path or VOCAB_PATH
        obj = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(obj)


def build_benzi_vocab_from_train() -> BenziVocab:
    """只看 train：收集出现过的本字。"""
    train = load_split("train")
    seen: set[str] = set()
    for ex in train:
        for b in ex["benzi"]:
            if b and b != BENZI_NONE:
                seen.add(b)
    itos = [BENZI_NONE, UNK] + sorted(seen)
    return BenziVocab(itos)


if __name__ == "__main__":
    vocab = build_benzi_vocab_from_train()
    path = vocab.save()
    print("vocab size:", len(vocab))
    print("saved:", path)
