# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the Apache License, Version 2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_size)
        # w1(x) -> (batch_size, seq_len, intermediate_size)
        # w3(x) -> (batch_size, seq_len, intermediate_size)
        # w2(*) -> (batch_size, seq_len, hidden_size)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, data_format="channels_first") -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.data_format = data_format

    def _norm(self, hidden_states):
        if self.data_format == "channels_first":
            # Calculate the mean along the height and width dimensions
            variance = hidden_states.pow(2).mean(dim=(1), keepdim=True)  # 在高和宽维度上计算均值
        elif self.data_format == "channels_last":
            variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)

    def forward(self, hidden_states):
        if self.data_format == "channels_first":
            return self.weight[..., None, None] * self._norm(hidden_states.float()).type_as(hidden_states)
        elif self.data_format == "channels_last":
            return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)
