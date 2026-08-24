#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BERT + BiLSTM + CRF（定位）+ 本字分类头；可选 BERT-LoRA。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from crf import CRF
from dataset import LABEL2ID
from lora_utils import apply_lora_to_bert, print_lora_summary
from torch_dataset import MODEL_NAME, TongjiaDataset, collate_batch

IGNORE_LABEL = -100


class BertBiLstmCrf(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = len(LABEL2ID),
        num_benzi: int = 0,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        benzi_loss_weight: float = 1.0,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
    ):
        super().__init__()
        self.benzi_loss_weight = float(benzi_loss_weight)
        self.use_lora = bool(use_lora)
        self.bert = AutoModel.from_pretrained(model_name)
        if self.use_lora:
            self.bert = apply_lora_to_bert(
                self.bert,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
        hidden = self.bert.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * lstm_hidden, num_labels)
        self.crf = CRF(num_labels)

        self.num_benzi = int(num_benzi)
        self.benzi_head = (
            nn.Linear(2 * lstm_hidden, self.num_benzi) if self.num_benzi > 0 else None
        )

    def print_param_summary(self) -> None:
        if self.use_lora:
            print_lora_summary(self)
        else:
            n = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"full finetune trainable params: {n:,}")

    def _hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        x, _ = self.lstm(x)
        return self.dropout(x)

    def _emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self._hidden(input_ids, attention_mask))

    @staticmethod
    def crf_mask(attention_mask: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        return attention_mask.bool() & (label_ids != IGNORE_LABEL)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_ids: torch.Tensor,
        benzi_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """训练：CRF loss（+ 可选本字 CE）。返回总 loss 标量。"""
        h = self._hidden(input_ids, attention_mask)
        emissions = self.classifier(h)
        mask = self.crf_mask(attention_mask, label_ids)
        tags = label_ids.clamp(min=0)
        crf_loss = self.crf(emissions, tags, mask)

        if self.benzi_head is None or benzi_ids is None or self.benzi_loss_weight <= 0:
            return crf_loss

        # 只在金标通假位置（B/I）上算本字损失
        tj_mask = (label_ids == 1) | (label_ids == 2)
        if not tj_mask.any():
            return crf_loss

        logits = self.benzi_head(h)
        benzi_loss = nn.functional.cross_entropy(
            logits[tj_mask],
            benzi_ids[tj_mask],
            ignore_index=IGNORE_LABEL,
        )
        return crf_loss + self.benzi_loss_weight * benzi_loss

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_ids: torch.Tensor | None = None,
    ) -> list[list[int]]:
        emissions = self.classifier(self._hidden(input_ids, attention_mask))
        if label_ids is None:
            mask = attention_mask.bool().clone()
            mask[:, 0] = False
            lengths = attention_mask.long().sum(dim=1)
            for b, L in enumerate(lengths.tolist()):
                if L > 1:
                    mask[b, L - 1] = False
        else:
            mask = self.crf_mask(attention_mask, label_ids)
        return self.crf.decode(emissions, mask)

    @torch.no_grad()
    def predict_benzi_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """本字分类 logits：[B, T, V]。"""
        if self.benzi_head is None:
            raise RuntimeError("模型没有本字头")
        h = self._hidden(input_ids, attention_mask)
        return self.benzi_head(h)

    @torch.no_grad()
    def predict_benzi_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """每个 token 的本字 id；无本字头时返回全 -100。"""
        if self.benzi_head is None:
            return torch.full(
                input_ids.shape, IGNORE_LABEL, dtype=torch.long, device=input_ids.device
            )
        pred = self.predict_benzi_logits(input_ids, attention_mask).argmax(dim=-1)
        if label_ids is not None:
            invalid = label_ids == IGNORE_LABEL
            pred = pred.masked_fill(invalid, IGNORE_LABEL)
        return pred


if __name__ == "__main__":
    from benzi_vocab import build_benzi_vocab_from_train

    print("Loading:", MODEL_NAME)
    vocab = build_benzi_vocab_from_train()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertBiLstmCrf(num_benzi=len(vocab), use_lora=True, lora_r=8)
    model.print_param_summary()
    model.eval()
    ds = TongjiaDataset("train", vocab)

    def _collate(batch):
        return collate_batch(batch, tokenizer)

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=_collate)
    batch = next(iter(loader))
    with torch.no_grad():
        loss = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["label_ids"],
            batch["benzi_ids"],
        )
    print("joint loss:", float(loss))
    print("num_benzi:", len(vocab))
