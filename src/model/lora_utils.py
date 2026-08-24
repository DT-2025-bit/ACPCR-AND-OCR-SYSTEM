#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BERT LoRA 包装（peft）。"""

from __future__ import annotations

from typing import Sequence

from peft import LoraConfig, PeftModel, get_peft_model


DEFAULT_TARGET_MODULES = ("query", "value")


def apply_lora_to_bert(
    bert,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Sequence[str] | None = None,
):
    """对 AutoModel 主干注入 LoRA；BiLSTM/CRF/本字头保持全参。"""
    targets = list(target_modules or DEFAULT_TARGET_MODULES)
    cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=targets,
        bias="none",
        modules_to_save=None,
    )
    return get_peft_model(bert, cfg)


def count_trainable_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_lora_summary(model) -> None:
    trainable, total = count_trainable_params(model)
    pct = 100.0 * trainable / max(total, 1)
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    bert = getattr(model, "bert", None)
    if isinstance(bert, PeftModel):
        bert.print_trainable_parameters()
