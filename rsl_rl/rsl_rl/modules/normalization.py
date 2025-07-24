#  Copyright (c) 2020 Preferred Networks, Inc.
#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn as nn


class Normalization(nn.Module):
    def __init__(self, shape, eps=1e-8):
        """Initialize Normalization module.
        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
        """
        super().__init__()
        
        self.eps = eps
        self.shape = shape
        
        self.register_buffer("_mean", torch.zeros(shape))
        self.register_buffer("_var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
    
    def forward(self, x):
        if self.training: self.update(x)
        num = x - self._mean[None]
        den = torch.sqrt(self._var[None] + self.eps)
        return num / den
    
    @torch.no_grad()
    def update(self, x):
        batch_size = x.shape[0]
        self.count += batch_size
        weight = batch_size / self.count
        
        mean_x = torch.mean(x, dim=0)
        delta_mean = mean_x - self._mean
        
        var_x = torch.var(x, dim=0, unbiased=False)
        real_var = torch.square(delta_mean) + var_x
        delta_var = real_var - self._var
        
        self._mean += weight * delta_mean
        self._var += weight * delta_var