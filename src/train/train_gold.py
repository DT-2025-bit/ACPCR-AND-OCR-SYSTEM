#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金标联合训练：定位（CRF）+ 本字分类。按 dev span-F1 存 best。

支持：
- 全参微调（use_lora: false）
- BERT-LoRA（use_lora: true）
- --init-ckpt 从银标预热权重继续金标精调
- 正例过采样 / 早停
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
from lexicon_filter import load_tongjia_benzi_map, load_tongjia_heads
from span_metrics import eval_joint_scores
from torch_dataset import TongjiaDataset, collate_batch
from train_utils import (
    build_model,
    eval_loss,
    load_checkpoint,
    load_config,
    make_tokenizer,
    resolve_model_name,
    save_checkpoint,
    set_seed,
)


def run_train(
    cfg: dict,
    *,
    init_ckpt: Path | None = None,
    phase_name: str = "gold",
) -> Path:
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = resolve_model_name(cfg["model_name"])
    print("=" * 60)
    print(f"phase: {phase_name}")
    print("device:", device)
    print("select best by: dev span-F1")
    print("model_name:", model_name)
    print("use_lora:", bool(cfg.get("use_lora", False)))

    lexicon_heads = load_tongjia_heads() if cfg.get("lexicon_filter", False) else None
    print(f"lexicon_filter: {'on' if lexicon_heads else 'off'}")
    tj_to_benzi = load_tongjia_benzi_map()
    print(f"benzi_lexicon_constrain: on  pairs={len(tj_to_benzi)}")

    if VOCAB_PATH.exists():
        benzi_vocab = BenziVocab.load()
    else:
        benzi_vocab = build_benzi_vocab_from_train()
        benzi_vocab.save()
    print("benzi vocab:", len(benzi_vocab))

    tokenizer = make_tokenizer(cfg)
    if init_ckpt is not None:
        model, ckpt_cfg, ckpt = load_checkpoint(init_ckpt, device=device)
        print(f"init from: {init_ckpt}  epoch={ckpt.get('epoch')}")
        model.benzi_loss_weight = float(cfg.get("benzi_loss_weight", 1.0))
        if bool(ckpt_cfg.get("use_lora", False)) != bool(cfg.get("use_lora", False)):
            raise ValueError("init_ckpt 的 use_lora 与当前配置不一致")
    else:
        model = build_model(cfg, num_benzi=len(benzi_vocab)).to(device)
    model.print_param_summary()

    def _collate(batch):
        return collate_batch(batch, tokenizer)

    pos_os = int(cfg.get("pos_oversample") or 1)
    print(f"pos_oversample: {pos_os}")
    train_ds = TongjiaDataset("train", benzi_vocab, pos_oversample=pos_os)
    print(f"train size after oversample: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=_collate,
    )
    dev_loader = DataLoader(
        TongjiaDataset("dev", benzi_vocab),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=_collate,
    )

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    out_dir = _PROJECT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    benzi_vocab.save(out_dir / "benzi_vocab.json")
    (out_dir / "config_used.yaml").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    max_steps = int(cfg.get("max_steps") or 0)
    grad_accum = max(int(cfg.get("grad_accum_steps") or 1), 1)
    patience = int(cfg.get("early_stop_patience") or 0)
    eval_max_batches = 3 if max_steps > 0 else 0
    use_amp = bool(cfg.get("fp16", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    best_f1 = -1.0
    bad_epochs = 0
    history: list[dict] = []

    model.train()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        pbar = tqdm(train_loader, desc=f"{phase_name} epoch {epoch}")
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

        if seen > 0 and step_i > 0 and (step_i % grad_accum != 0):
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        train_loss = running / max(seen, 1)
        dev_loss = eval_loss(model, dev_loader, device, max_batches=eval_max_batches)
        scores = eval_joint_scores(
            model,
            dev_loader,
            device,
            max_batches=eval_max_batches,
            lexicon_heads=lexicon_heads,
            tj_to_benzi=tj_to_benzi,
            benzi_stoi=benzi_vocab.stoi,
        )
        dev_f1 = scores["f1"]
        print(
            f"epoch {epoch}  train_loss={train_loss:.4f}  "
            f"dev_loss={dev_loss:.4f}  "
            f"dev_P={scores['precision']:.4f}  "
            f"dev_R={scores['recall']:.4f}  "
            f"dev_F1={dev_f1:.4f}  "
            f"char_F1={scores.get('char_f1', 0):.4f}  "
            f"top50_F1={scores.get('top50_f1', 0):.4f}  "
            f"benzi_oracle={scores['benzi_acc_oracle']:.4f}"
        )
        history.append(
            {
                "phase": phase_name,
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                **{k: scores[k] for k in scores},
            }
        )

        metrics = {
            "dev_loss": dev_loss,
            "dev_f1": dev_f1,
            "dev_precision": scores["precision"],
            "dev_recall": scores["recall"],
            "dev_char_f1": scores.get("char_f1"),
            "dev_top50_f1": scores.get("top50_f1"),
            "benzi_acc_oracle": scores["benzi_acc_oracle"],
        }
        save_checkpoint(
            out_dir / "last.pt",
            model,
            cfg,
            epoch=epoch,
            num_benzi=len(benzi_vocab),
            metrics=metrics,
        )
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            bad_epochs = 0
            save_checkpoint(
                out_dir / "best.pt",
                model,
                cfg,
                epoch=epoch,
                num_benzi=len(benzi_vocab),
                metrics=metrics,
            )
            print(f"  saved best.pt (dev_F1={dev_f1:.4f})")
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"early stop: no improve for {patience} epochs.")
                break

        if max_steps > 0 and global_step >= max_steps:
            print(f"reached max_steps={max_steps}, stop.")
            break

    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("done. best_dev_F1=", f"{best_f1:.4f}", "out_dir:", out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(_SRC / "configs" / "train_gold_v2.yaml"),
    )
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default="",
        help="可选：从银标预热或其它 checkpoint 继续金标精调",
    )
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    init = Path(args.init_ckpt) if args.init_ckpt else None
    if init is None and cfg.get("init_ckpt"):
        init = _PROJECT / str(cfg["init_ckpt"])
    run_train(cfg, init_ckpt=init, phase_name="gold")


if __name__ == "__main__":
    main()
