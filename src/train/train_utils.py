#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练/评测共用：配置、建模、存盘、加载。"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

_SRC = Path(__file__).resolve().parents[1]
_PROJECT = _SRC.parent
for p in (_SRC, _SRC / "data", _SRC / "model", _SRC / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bert_bilstm_crf import BertBiLstmCrf
from dataset import LABEL2ID


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_name(name: str, project: Path | None = None) -> str:
    project = project or _PROJECT
    p = Path(name)
    if p.exists():
        return str(p)
    cand = project / name
    if cand.exists():
        return str(cand)
    return name


def build_model(cfg: dict, num_benzi: int, model_name: str | None = None) -> BertBiLstmCrf:
    name = resolve_model_name(model_name or cfg["model_name"])
    targets = cfg.get("lora_target_modules")
    return BertBiLstmCrf(
        model_name=name,
        num_labels=len(LABEL2ID),
        num_benzi=num_benzi,
        lstm_hidden=int(cfg["lstm_hidden"]),
        lstm_layers=int(cfg["lstm_layers"]),
        dropout=float(cfg["dropout"]),
        benzi_loss_weight=float(cfg.get("benzi_loss_weight", 1.0)),
        use_lora=bool(cfg.get("use_lora", False)),
        lora_r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        lora_target_modules=list(targets) if targets else None,
    )


def load_checkpoint(
    ckpt_path: Path,
    device: torch.device | None = None,
) -> tuple[BertBiLstmCrf, dict, dict]:
    """返回 (model, cfg, ckpt_meta)。"""
    device = device or torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    num_benzi = int(ckpt.get("num_benzi") or 0)
    model = build_model(cfg, num_benzi=num_benzi)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, cfg, ckpt


def save_checkpoint(
    path: Path,
    model: BertBiLstmCrf,
    cfg: dict,
    *,
    epoch: int,
    num_benzi: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "num_benzi": num_benzi,
        "use_lora": bool(cfg.get("use_lora", False)),
    }
    if metrics:
        payload.update(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


@torch.no_grad()
def eval_loss(model, loader, device, max_batches: int = 0) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        loss = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["label_ids"].to(device),
            batch["benzi_ids"].to(device),
        )
        total += float(loss.detach()) * batch["input_ids"].size(0)
        n += batch["input_ids"].size(0)
        if max_batches > 0 and (i + 1) >= max_batches:
            break
    model.train()
    return total / max(n, 1)


def make_tokenizer(cfg: dict):
    return AutoTokenizer.from_pretrained(resolve_model_name(cfg["model_name"]))
