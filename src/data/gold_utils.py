"""ACPCR 金标处理的公共工具。

本模块被 process_gold.py / validate_gold.py 共用，负责：
1. 路径常量（原料目录、输出目录）
2. 字头清洗、位置解析、字对 ID
3. JSONL 读写
4. span ↔ BIO 转换与单条样本校验
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# 路径：原料在 Golden Label Data，加工结果写到 processed/gold
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_RAW = PROJECT_ROOT / "data" / "Golden Label Data" # 通假字资源库
GOLD_OUT = PROJECT_ROOT / "data" / "processed" / "gold" # 加工后的金标数据

CORPUS_PATH = GOLD_RAW / "corpus" / "corpus.jsonl" # 专家语料库
LEXICON_PATH = GOLD_RAW / "knowledge_base" / "tongjia_links.jsonl" # 规范表
EVAL_DIR = GOLD_RAW / "evaluation" / "tongjiazi_evaluation" # 检测/识别评测集，用来补负例

# 《汉语大词典》字头常带义项序号，如「耗3」「眊1」→ 训练时只需字形
HEAD_SUFFIX_RE = re.compile(r"\d+$")


def strip_head_suffix(head: str) -> str:
    """去掉字头末尾数字义项号，例如「耗3」→「耗」。"""
    head = (head or "").strip()
    if not head:
        return head
    return HEAD_SUFFIX_RE.sub("", head)


def parse_positions(raw: str | int) -> list[int]:
    """解析原料里的「标注位置」。

    原料可能是整数，也可能是「6, 13」这类一句可以有多个通假。
    """
    if isinstance(raw, int):
        return [raw]
    text = str(raw).strip()
    if not text:
        return []
    parts = re.split(r"[,，\s]+", text)
    return [int(p) for p in parts if p]


def pair_id(tongjia: str, benzi: str, relation_id: int | None = None) -> str:
    """生成通假字对稳定 ID，便于分层划分与统计。

    例：TJ_耗_眊_0000
    """
    suffix = f"_{relation_id:04d}" if relation_id is not None else ""
    return f"TJ_{tongjia}_{benzi}{suffix}"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL（每行一个 JSON 对象）。"""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """写出 JSONL；ensure_ascii=False 保留中文原文。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def spans_to_bio(text: str, spans: list[dict[str, Any]]) -> list[str]:
    """把字符级 span 转成 BIO 标签序列（训练主任务用）。

    - O：非通假
    - B-TJ：通假片段首字
    - I-TJ：通假片段非首字（连续多字时）
    """
    labels = ["O"] * len(text)
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    for span in ordered:
        start, end = span["start"], span["end"]
        if start < 0 or end > len(text) or start >= end:
            raise ValueError(f"invalid span {start}:{end} for len={len(text)}")
        labels[start] = "B-TJ"
        for i in range(start + 1, end):
            if labels[i] != "O":
                raise ValueError(f"overlapping span at {i}")
            labels[i] = "I-TJ"
    return labels


def validate_record(rec: dict[str, Any]) -> list[str]:
    """单条金标样本的结构校验，返回错误列表（空表示通过）。

    检查项对齐规范草案：边界合法、无重叠、span 文本与偏移一致、可转 BIO。
    """
    errors: list[str] = []
    text = rec.get("text", "")
    spans = rec.get("spans", [])
    if not isinstance(text, str) or not text:
        errors.append("empty text")
        return errors
    if not isinstance(spans, list):
        errors.append("spans not list")
        return errors

    # 同一 (start, end) 不允许重复
    seen: set[tuple[int, int]] = set()
    for idx, span in enumerate(spans):
        start, end = span.get("start"), span.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"span#{idx} start/end not int")
            continue
        # 右开区间 [start, end)
        if not (0 <= start < end <= len(text)):
            errors.append(f"span#{idx} out of bounds: {start}:{end}")
            continue
        key = (start, end)
        if key in seen:
            errors.append(f"duplicate span {start}:{end}")
        seen.add(key)
        # 偏移切出的字必须与标注的通假字一致
        chunk = text[start:end]
        if span.get("tongjia") and chunk != span["tongjia"]:
            errors.append(f"span#{idx} text mismatch: '{chunk}' != '{span['tongjia']}'")

    # 字符级不可重叠（禁止两个 span 覆盖同一字符）
    occupied: set[int] = set()
    for span in spans:
        for i in range(span["start"], span["end"]):
            if i in occupied:
                errors.append(f"overlapping span at index {i}")
            occupied.add(i)

    # 能无损转成 BIO 才入库
    if spans:
        try:
            spans_to_bio(text, spans)
        except ValueError as exc:
            errors.append(str(exc))

    return errors