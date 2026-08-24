# ACPCR：古汉语通假字定位与本字预测

面向**数据分析 / NLP 应用**场景的端到端小项目：在古汉语句子中自动标出通假字位置，并给出本字候选。

强调点不只是调模型，还包括：**数据质检决策、基线对照、消融对比、可复现评测与 Demo**。

## 做了什么

1. **金标流水线**：通假资源库 → 统一 JSONL / BIO，train·dev·test 划分与校验  
2. **银标质检**：词典匹配候选 → 规则过滤 → 分层抽检 → 结论「正例不可训、负例可作 warmup」  
3. **模型**：`chinese-bert-wwm-ext` + BiLSTM + CRF（定位）+ 本字分类头；支持全参 / LoRA  
4. **推理约束**：规范表过滤误报 span；本字仅在规范表候选中选取  
5. **评测与消融**：词典基线 vs 全参 vs LoRA vs 银标负例预热（主指标只认金标 test）

## 主要结果（金标 test）

| 设定 | span-F1 | Top-50 F1 | P | R | 本字 Acc（oracle） |
|------|--------:|----------:|--:|--:|-------------------:|
| 词典匹配基线 | 0.134 | 0.164 | 0.072 | 1.000 | 0.650 |
| 金标全参 v1 | 0.493 | 0.856 | 0.602 | 0.418 | 0.635 |
| **主模型：金标全参 v2** | **0.546** | **0.875** | 0.580 | 0.516 | 0.653 |
| 金标 LoRA | 0.300 | 0.613 | 0.629 | 0.197 | 0.658 |
| 全参 + 银标负例预热 | 0.468 | 0.844 | 0.636 | 0.370 | 0.631 |

- **span-F1**：全量 test，严格边界（主指标）；长尾字对会拉低分数  
- **Top-50 F1**：train 上最高频 50 个通假字对（阅读辅助更关心的头部分布）；主模型约 **0.16 → 0.88**  
- 银标伪标正例抽检可用率约 **2%**，故**不参与主模型训练**；负例仅用于轻量 warmup 消融  

指标 JSON 见 `checkpoints/*/metrics_test.json`（字段含 `f1` / `top50_f1`；`.pt` 权重默认不提交）。

## 数据说明（仓库不附带语料本体）

版权与体积原因，**金标 / 银标原文与 processed 大文件不入库**；用本地数据 + `src/data/` 脚本复现即可。

| 层级 | 规模（本机实验） | 用途 |
|------|------------------|------|
| **金标** | 约 1.85 万句（train/dev/test ≈ 8:1:1） | 主训练与**唯一对外主评测** |
| **银标** | 远监督候选 + 分层抽检 | 质检结论：正例约 2% 可用 → **不参训**；负例约 1.1 万导出作 warmup 消融 |

相关脚本：`process_gold.py` / `validate_gold.py`、`match_silver_candidates.py` / `filter_silver.py` / `spotcheck_silver.py` / `export_silver_warmup.py`。

## 环境

- Python ≥ 3.11，推荐用 [uv](https://github.com/astral-sh/uv)  
- GPU + CUDA 版 PyTorch（CPU 可跑通，训练较慢）

```powershell
cd GitHub_Release   # 或你的克隆目录
uv sync
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 可选，国内下载更快
```

训练前需要中文 BERT 预训练权重（考虑到体积较大，不放进 Git）。任选其一：

- 配置里写 Hub 名：`hfl/chinese-bert-wwm-ext`（首次运行自动下载），或  
- 下载到本地目录，再在 `src/configs/*.yaml` 的 `model_name` 填本地路径  

## 快速使用

```powershell
# 词典基线（无需训练）
uv run python src/eval/eval_lexicon_baseline.py --split test

# 训练主模型（示例）
uv run python src/train/train_gold.py --config src/configs/train_gold_v2.yaml

# 评测
uv run python src/eval/eval_span_f1.py --ckpt checkpoints/gold_full_v2/best.pt --split test

# Demo
uv run python src/demo/predict_cli.py --text "塗金披繡，漿酒藿肉者，故不可稱紀。"
```

更完整说明见各脚本 `--help` 与 `src/configs/`。

## 目录结构

```text
├── src/             # 数据脚本 / 模型 / 训练 / 评测 / Demo
├── checkpoints/     # 默认仅保留 metrics 等小文件（无 .pt）
├── pyproject.toml
└── uv.lock
```

## 简历可写的项目要点（建议）

- 通假字 **定位 + 本字** 双任务；金标约 1.8 万句可复现划分  
- 银标抽检驱动训练策略（拒绝脏正例），体现数据质量意识  
- 相对规则基线有明确增益，并报告全参 / LoRA / 预热消融  
- 提供 CLI Demo 与冻结 test 评测脚本  

## 许可与数据

- 代码与配置：可按需自选开源许可（如 MIT）  
- 金标 / 银标原料请遵循原始数据方的许可与引用要求；大语料与权重默认不随仓库分发  
