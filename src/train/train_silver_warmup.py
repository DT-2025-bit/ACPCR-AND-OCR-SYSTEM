#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""银标负例轻量预热（1 epoch）→ 再金标多任务精调。

对应任务书阶段 2 / 消融 A1：
  仅金标 vs 金标 + 银标子采样预热
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1]
_PROJECT = _SRC.parent
for p in (_SRC, _SRC / "data", _SRC / "model", _SRC / "eval", _SRC / "train"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benzi_vocab import BenziVocab, VOCAB_PATH, build_benzi_vocab_from_train
from torch_dataset import SILVER_WARMUP_PATH, TongjiaDataset, collate_batch
from train_gold import run_train
from train_utils import (
    build_model,
    load_config,
    make_tokenizer,
    resolve_model_name,
    save_checkpoint,
    set_seed,
)


def run_silver_warmup(cfg: dict) -> Path:
    """阶段 A：仅定位（负例全 O），默认 1 epoch。"""
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = resolve_model_name(cfg["model_name"])
    silver_path = Path(cfg.get("silver_path") or SILVER_WARMUP_PATH)
    if not silver_path.is_absolute():
        silver_path = _PROJECT / silver_path
    if not silver_path.exists():
        raise FileNotFoundError(f"银标预热包不存在: {silver_path}")

    print("=" * 60)
    print("phase: silver_warmup")
    print("device:", device)
    print("model_name:", model_name)
    print("silver_path:", silver_path)
    print("use_lora:", bool(cfg.get("use_lora", False)))

    if VOCAB_PATH.exists():
        benzi_vocab = BenziVocab.load()
    else:
        benzi_vocab = build_benzi_vocab_from_train()
        benzi_vocab.save()

    # 预热阶段关闭本字损失（负例无通假位）
    warm_cfg = dict(cfg)
    warm_cfg["benzi_loss_weight"] = float(cfg.get("warmup_benzi_loss_weight", 0.0))

    tokenizer = make_tokenizer(cfg)
    model = build_model(warm_cfg, num_benzi=len(benzi_vocab)).to(device)
    model.print_param_summary()

    def _collate(batch):
        return collate_batch(batch, tokenizer)

    train_loader = DataLoader(
        TongjiaDataset(benzi_vocab=benzi_vocab, jsonl_path=silver_path),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=_collate,
    )

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg.get("warmup_lr", cfg["lr"])),
        weight_decay=float(cfg["weight_decay"]),
    )

    out_dir = _PROJECT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    warmup_dir = out_dir / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    benzi_vocab.save(out_dir / "benzi_vocab.json")
    benzi_vocab.save(warmup_dir / "benzi_vocab.json")

    epochs = int(cfg.get("warmup_epochs", 1))
    max_steps = int(cfg.get("warmup_max_steps") or 0)
    grad_accum = max(int(cfg.get("grad_accum_steps") or 1), 1)
    use_amp = bool(cfg.get("fp16", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    history: list[dict] = []
    model.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"silver warmup epoch {epoch}")
        running, seen = 0.0, 0
        optim.zero_grad(set_to_none=True)
        step_i = 0

        for step_i, batch in enumerate(pbar, start=1):
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["label_ids"].to(device),
                    batch["benzi_ids"].to(device),
                )
                loss = loss / grad_accum
            scaler.scale(loss).backward()

            if step_i % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            bs = batch["input_ids"].size(0)
            running += float(loss.detach()) * grad_accum * bs
            seen += bs
            global_step += 1
            pbar.set_postfix(loss=f"{float(loss.detach()) * grad_accum:.4f}")

            if max_steps > 0 and global_step >= max_steps:
                break

        if step_i > 0 and (step_i % grad_accum != 0):
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        train_loss = running / max(seen, 1)
        print(f"warmup epoch {epoch}  train_loss={train_loss:.4f}  n={seen}")
        history.append(
            {
                "phase": "silver_warmup",
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "n": seen,
            }
        )

        save_checkpoint(
            warmup_dir / "last.pt",
            model,
            warm_cfg,
            epoch=epoch,
            num_benzi=len(benzi_vocab),
            metrics={"train_loss": train_loss},
        )
        save_checkpoint(
            warmup_dir / "best.pt",
            model,
            warm_cfg,
            epoch=epoch,
            num_benzi=len(benzi_vocab),
            metrics={"train_loss": train_loss},
        )

        if max_steps > 0 and global_step >= max_steps:
            break

    (warmup_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("warmup done →", warmup_dir / "best.pt")
    return warmup_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(_SRC / "configs" / "train_silver_warmup.yaml"),
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="跳过预热，仅用 config.init_ckpt / warmup/best.pt 做金标精调",
    )
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    if args.skip_warmup:
        init = Path(cfg.get("init_ckpt") or "")
        if not init.is_absolute():
            init = _PROJECT / init
        if not init.exists():
            init = _PROJECT / cfg["out_dir"] / "warmup" / "best.pt"
    else:
        init = run_silver_warmup(cfg)

    # 阶段 B：金标精调（恢复本字损失）
    gold_cfg = dict(cfg)
    gold_cfg["benzi_loss_weight"] = float(cfg.get("benzi_loss_weight", 1.0))
    gold_cfg["epochs"] = int(cfg.get("gold_epochs", cfg["epochs"]))
    gold_cfg["lr"] = float(cfg.get("gold_lr", cfg["lr"]))
    run_train(gold_cfg, init_ckpt=init, phase_name="gold_finetune")


if __name__ == "__main__":
    main()
