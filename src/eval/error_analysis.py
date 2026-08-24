#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金标 test 错误分析：导出漏标 / 误标 / 本字错样本。"""

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

from benzi_vocab import BenziVocab
from dataset import ID2LABEL, BENZI_NONE
from lexicon_filter import (
    constrain_benzi_by_lexicon,
    filter_pred_ids_by_lexicon,
    load_tongjia_benzi_map,
    load_tongjia_heads,
    tags_to_spans,
)
from torch_dataset import TongjiaDataset, collate_batch
from train_utils import load_checkpoint, make_tokenizer

IGNORE_LABEL = -100


def spans_as_set(tags: list[str]) -> set[tuple[int, int]]:
    return set(tags_to_spans(tags))


def highlight(chars: list[str], spans: set[tuple[int, int]], mark: str = "【】") -> str:
    left, right = mark[0], mark[1]
    out = []
    i = 0
    n = len(chars)
    span_starts = {s: e for s, e in spans}
    while i < n:
        if i in span_starts:
            e = span_starts[i]
            out.append(left + "".join(chars[i:e]) + right)
            i = e
        else:
            out.append(chars[i])
            i += 1
    return "".join(out)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=str(_PROJECT / "checkpoints" / "gold_full_v2" / "best.pt"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit-each", type=int, default=15)
    parser.add_argument(
        "--out",
        default=str(_PROJECT / "docs" / "error_samples.jsonl"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(Path(args.ckpt), device=device)
    model.eval()
    vocab = BenziVocab.load(Path(args.ckpt).parent / "benzi_vocab.json")
    tok = make_tokenizer(cfg)
    heads = load_tongjia_heads() if cfg.get("lexicon_filter", True) else None
    tj_map = load_tongjia_benzi_map()

    loader = DataLoader(
        TongjiaDataset(args.split, vocab),
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tok),
    )

    misses: list[dict] = []
    falses: list[dict] = []
    benzi_errs: list[dict] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_ids = batch["label_ids"]
        preds = model.decode(input_ids, attention_mask, label_ids.to(device))
        logits = model.predict_benzi_logits(input_ids, attention_mask)

        for bi in range(input_ids.size(0)):
            if (
                len(misses) >= args.limit_each
                and len(falses) >= args.limit_each
                and len(benzi_errs) >= args.limit_each
            ):
                break

            chars = batch["chars"][bi]
            gold_ids = [x for x in label_ids[bi].tolist() if x != IGNORE_LABEL]
            n = min(len(chars), len(gold_ids), len(preds[bi]))
            chars = chars[:n]
            gold_tags = [ID2LABEL[i] for i in gold_ids[:n]]
            p_ids = preds[bi][:n]
            if heads is not None:
                p_ids = filter_pred_ids_by_lexicon(p_ids, chars, heads)
            pred_tags = [ID2LABEL[i] for i in p_ids]

            gset = spans_as_set(gold_tags)
            pset = spans_as_set(pred_tags)
            sid = batch["ids"][bi]
            text = "".join(chars)

            for sp in sorted(gset - pset):
                if len(misses) < args.limit_each:
                    misses.append(
                        {
                            "type": "E_MISS",
                            "id": sid,
                            "text": text,
                            "gold_span": list(sp),
                            "surface": "".join(chars[sp[0] : sp[1]]),
                            "gold_view": highlight(chars, gset),
                            "pred_view": highlight(chars, pset),
                        }
                    )
            for sp in sorted(pset - gset):
                if len(falses) < args.limit_each:
                    falses.append(
                        {
                            "type": "E_FALSE",
                            "id": sid,
                            "text": text,
                            "pred_span": list(sp),
                            "surface": "".join(chars[sp[0] : sp[1]]),
                            "gold_view": highlight(chars, gset),
                            "pred_view": highlight(chars, pset),
                        }
                    )

            # 本字：仅定位都命中的 span
            keep = label_ids[bi] != IGNORE_LABEL
            char_logits = logits[bi][keep][:n]
            p_bz = constrain_benzi_by_lexicon(char_logits, chars, tj_map, vocab.stoi)
            g_bz_ids = [x for x in batch["benzi_ids"][bi].tolist() if x != IGNORE_LABEL][:n]
            for s, e in sorted(gset & pset):
                if len(benzi_errs) >= args.limit_each:
                    break
                gb = g_bz_ids[s] if s < len(g_bz_ids) else -1
                pb = p_bz[s] if s < len(p_bz) else -1
                if gb < 0 or pb < 0:
                    continue
                if gb == pb:
                    continue
                g_str = vocab.decode(gb)
                p_str = vocab.decode(pb)
                if g_str in (BENZI_NONE, "[UNK]"):
                    continue
                benzi_errs.append(
                    {
                        "type": "E_BENZI",
                        "id": sid,
                        "text": text,
                        "span": [s, e],
                        "surface": "".join(chars[s:e]),
                        "gold_benzi": g_str,
                        "pred_benzi": p_str,
                        "gold_view": highlight(chars, gset),
                        "pred_view": highlight(chars, pset),
                    }
                )

    rows = misses + falses + benzi_errs
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    md_path = _PROJECT / "docs" / "错误分析.md"
    lines = [
        "# 错误分析（主模型 gold_full_v2 / test）",
        "",
        f"样本由 `src/eval/error_analysis.py` 自动抽取（每类最多 {args.limit_each} 条）。",
        "",
        "## 1. 统计口径",
        "",
        "| 类型 | 含义 | 本批条数 |",
        "| --- | --- | ---: |",
        f"| E_MISS | 金标有、预测无（漏标） | {len(misses)} |",
        f"| E_FALSE | 预测有、金标无（误标） | {len(falses)} |",
        f"| E_BENZI | 定位命中但本字错 | {len(benzi_errs)} |",
        "",
        "## 2. 现象归纳",
        "",
        "1. **漏标（E_MISS）**：全量召回约 0.52，长尾字对与弱上下文易漏；高频头召回明显更高。",
        "2. **误标（E_FALSE）**：规范表字头在句中为本用时仍可能被标（难负例场景）；lexicon_filter 只能去掉「非字头」误报。",
        "3. **本字错（E_BENZI）**：一通多本时取错候选；oracle Acc≈0.65，定位命中位上约 0.73。",
        "",
        "## 3. 样例（摘录）",
        "",
        "### 3.1 漏标",
        "",
    ]
    for r in misses[:5]:
        lines.append(f"- `{r['id']}` 漏「{r['surface']}」")
        lines.append(f"  - 金标：{r['gold_view']}")
        lines.append(f"  - 预测：{r['pred_view']}")
        lines.append("")
    lines.extend(["### 3.2 误标", ""])
    for r in falses[:5]:
        lines.append(f"- `{r['id']}` 误报「{r['surface']}」")
        lines.append(f"  - 金标：{r['gold_view']}")
        lines.append(f"  - 预测：{r['pred_view']}")
        lines.append("")
    lines.extend(["### 3.3 本字错", ""])
    for r in benzi_errs[:5]:
        lines.append(
            f"- `{r['id']}` 「{r['surface']}」金标本字={r['gold_benzi']} / 预测={r['pred_benzi']}"
        )
        lines.append(f"  - {r['gold_view']}")
        lines.append("")
    lines.extend(
        [
            "## 4. 改进方向（不作为本期必达）",
            "",
            "- 针对 E_MISS：难例补标 / 正例再加权（v2 已做一层过采样）",
            "- 针对 E_FALSE：加强难负例或上下文规则",
            "- 针对 E_BENZI：本字候选消歧、多义项置信度",
            "",
            f"原始 JSONL：`{out.as_posix()}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"miss={len(misses)} false={len(falses)} benzi={len(benzi_errs)}")
    print("wrote:", out)
    print("wrote:", md_path)


if __name__ == "__main__":
    main()
