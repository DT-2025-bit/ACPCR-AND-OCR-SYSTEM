#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""线性链 CRF：给每个位置的标签打分，再学合法的标签转移。

支持 mask 中间一段为 True（BERT：CLS/SEP/PAD=False，汉字=True）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRF(nn.Module):
    """
    emissions: [B, T, C]  每个位置对 C 类标签的发射分
    tags:      [B, T]     金标（mask 为 True 的位置必须是 0..C-1，不能是 -100）
    mask:      [B, T]     True/1 = 有效字；False/0 = PAD / CLS / SEP 等
    """

    def __init__(self, num_tags: int):
        super().__init__()
        if num_tags <= 0:
            raise ValueError("num_tags 必须 > 0")
        self.num_tags = num_tags

        # transitions[i, j] = 从上一个标签 i 转到当前标签 j 的分数
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """训练：返回 batch 平均的负对数似然（越小越好，可直接当 loss）。"""
        self._validate(emissions, tags=tags, mask=mask)
        mask = mask.bool()
        if not mask.any():
            raise ValueError("mask 不能全为 False")

        gold_score = self._compute_gold_score(emissions, tags, mask)
        log_Z = self._compute_log_partition(emissions, mask)
        nll = log_Z - gold_score
        return nll.mean()

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """推理：Viterbi，只返回 mask=True 位置的标签。"""
        self._validate(emissions, mask=mask)
        mask = mask.bool()
        return self._viterbi_decode(emissions, mask)

    @staticmethod
    def _starts_ends(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """每条样本第一个/最后一个 True 的下标。"""
        batch_size, seq_len = mask.shape
        idx = torch.arange(seq_len, device=mask.device).unsqueeze(0).expand(batch_size, seq_len)
        # 无效处置成极大/极小，便于 min/max
        starts = torch.where(mask, idx, torch.full_like(idx, seq_len)).min(dim=1).values
        ends = torch.where(mask, idx, torch.full_like(idx, -1)).max(dim=1).values
        return starts, ends

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """金标路径的总分 score(y)。"""
        batch_size, seq_len = tags.shape
        score = emissions.new_zeros(batch_size)
        starts, ends = self._starts_ends(mask)

        # 第一个有效位置：start + emit
        first_tags = tags.gather(1, starts.unsqueeze(1)).squeeze(1)
        score = score + self.start_transitions[first_tags]
        first_emit = emissions.gather(
            1, starts.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_tags)
        ).squeeze(1)
        score = score + first_emit.gather(1, first_tags.unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            # 仅当 t 有效、且不是该样本起点时，才加「转移 + 发射」
            m = mask[:, t] & (starts < t)
            if not m.any():
                continue
            prev = tags[:, t - 1]
            curr = tags[:, t]
            emit = emissions[:, t].gather(1, curr.unsqueeze(1)).squeeze(1)
            trans = self.transitions[prev, curr]
            score = score + (emit + trans) * m.to(emissions.dtype)

        last_tags = tags.gather(1, ends.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """所有可能路径的 log Z。"""
        batch_size, seq_len, num_tags = emissions.shape
        starts, _ends = self._starts_ends(mask)

        # 尚未开始的样本用 -inf，避免污染
        neg_inf = -1e4
        alpha = emissions.new_full((batch_size, num_tags), neg_inf)

        for t in range(seq_len):
            emit = emissions[:, t]  # [B, C]
            is_start = (starts == t).unsqueeze(1)  # [B, 1]
            m = mask[:, t].unsqueeze(1)

            start_score = self.start_transitions + emit
            if t == 0:
                next_from_prev = start_score  # 占位，下面用 where
            else:
                next_from_prev = torch.logsumexp(
                    alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emit.unsqueeze(1),
                    dim=1,
                )

            # 起点用 start+emit；后续有效步用转移；无效步保持 alpha
            stepped = torch.where(is_start, start_score, next_from_prev)
            alpha = torch.where(m, stepped, alpha)

        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        batch_size, seq_len, num_tags = emissions.shape
        starts, ends = self._starts_ends(mask)

        neg_inf = -1e4
        score = emissions.new_full((batch_size, num_tags), neg_inf)
        # history[t-1]: 到达时刻 t 时，每个标签的前驱（仅 t>=1 有意义）
        history: list[torch.Tensor] = []

        for t in range(seq_len):
            emit = emissions[:, t]
            is_start = (starts == t).unsqueeze(1)
            m = mask[:, t].unsqueeze(1)

            start_score = self.start_transitions + emit
            if t == 0:
                next_score = start_score
                indices = torch.zeros(batch_size, num_tags, dtype=torch.long, device=emissions.device)
            else:
                cand = score.unsqueeze(2) + self.transitions + emit.unsqueeze(1)
                next_score, indices = cand.max(dim=1)

            stepped = torch.where(is_start, start_score, next_score)
            score = torch.where(m, stepped, score)
            if t > 0:
                history.append(indices)

        score = score + self.end_transitions
        _best_last_scores, best_last_tags = score.max(dim=1)

        best_paths: list[list[int]] = []
        for b in range(batch_size):
            start = int(starts[b].item())
            end = int(ends[b].item())
            tag = int(best_last_tags[b].item())
            path = [tag]
            # history[k] 对应时刻 k+1；从 end 回溯到 start+1
            for t in range(end, start, -1):
                # history[t-1] 是到达 t 的前驱
                tag = int(history[t - 1][b, tag].item())
                path.append(tag)
            path.reverse()
            best_paths.append(path)
        return best_paths

    def _validate(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        tags: torch.Tensor | None = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError("emissions 应为 [B, T, C]")
        if emissions.size(2) != self.num_tags:
            raise ValueError("emissions 最后一维必须等于 num_tags")
        if mask.shape != emissions.shape[:2]:
            raise ValueError("mask 应为 [B, T]")
        if tags is not None and tags.shape != emissions.shape[:2]:
            raise ValueError("tags 应为 [B, T]")


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 2, 6, 3
    crf = CRF(C)
    emissions = torch.randn(B, T, C, requires_grad=True)
    # 模拟 BERT：位置0=CLS、末有效后=SEP/PAD → mask 中间为 True
    tags = torch.tensor(
        [
            [0, 0, 1, 2, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )
    mask = torch.tensor(
        [
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    loss = crf(emissions, tags, mask)
    loss.backward()
    pred = crf.decode(emissions.detach(), mask)
    print("loss:", float(loss.detach()))
    print("decode:", pred)
    print("pred lens (expect 3, 2):", [len(p) for p in pred])
