# Demo

```powershell
# 单句
uv run python src/demo/predict_cli.py --text "塗金披繡，漿酒藿肉者，故不可稱紀。"

# 交互（直接回车结束）
uv run python src/demo/predict_cli.py

# 指定 checkpoint
uv run python src/demo/predict_cli.py --ckpt checkpoints/gold_full_v2/best.pt --text "……"
```

默认加载主模型 `checkpoints/gold_full_v2/best.pt`。输出含高亮串与 span 列表（通假→本字）。
