#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Scaler(nn.Module):
    def __init__(self, input_dim, init=1.0, scale=1.0):
        super().__init__()
        self.forward_scaler = init / scale
        self.scaler = nn.Parameter(
            torch.zeros(input_dim) + 1.0 * scale)
        
    def forward(self, x):
        return self.scaler * self.forward_scaler * x
    

class HyperLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, init=1.0, scale=1.0, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.scaler = Scaler(hidden_dim, init, scale)
        nn.init.orthogonal_(self.w1.weight)
        nn.init.orthogonal_(self.w2.weight)
        
    def forward(self, x):
        x = self.w1(x)
        x = F.relu(self.scaler(x)) + self.eps
        x = self.w2(x)
        x = F.normalize(x, p=2.0, dim=-1, eps=self.eps)
        return x
    

class HyperEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, init=1.0, scale=1.0, c_shift=3.0, eps=1e-8):
        super().__init__()
        self.w = nn.Linear(input_dim + 1, hidden_dim, bias=False)
        self.scaler = Scaler(hidden_dim, init, scale)
        self.c_shift, self.eps = c_shift, eps
        nn.init.orthogonal_(self.w.weight)

    def forward(self, x):
        new_axis = torch.zeros_like(x[..., :1]) + self.c_shift
        x = torch.cat([x, new_axis], dim=-1)
        x = F.normalize(x, p=2.0, dim=-1, eps=self.eps)
        x = self.w(x)
        x = self.scaler(x)
        x = F.normalize(x, p=2.0, dim=-1, eps=self.eps)
        return x
    

class HyperLERPBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim,
        expansion=2, 
        scaler_init=1.0,
        scaler_scale=1.0,
        alpha_init=1.0,
        alpha_scale=1.0, 
        eps=1e-8
        ):
        super().__init__()
        assert isinstance(expansion, int)
        self.eps = eps
        self.hyper_layer = HyperLayer(
            hidden_dim,
            hidden_dim * expansion,
            hidden_dim, 
            scaler_init / math.sqrt(expansion),
            scaler_scale / math.sqrt(expansion),
            eps=eps,
            )
        self.hyper_scaler = Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x):
        residual = x
        x = self.hyper_layer(x)
        x = residual + self.hyper_scaler(x - residual)
        x = F.normalize(x, p=2.0, dim=-1, eps=self.eps)
        return x


class HyperPolicyHead(nn.Module):
    def __init__(self, hidden_dim, action_dim, init=1.0, scale=1.0):
        super().__init__()
        self.mean_w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mean_scaler = Scaler(hidden_dim, init, scale)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.mean_w1.weight)
        
    def forward(self, x):
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_head(mean)
        return mean


class HyperValueHead(nn.Module):
    def __init__(self, hidden_dim, init=1.0, scale=1.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scaler = Scaler(hidden_dim, init, scale)
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.w1.weight)

    def forward(self, x):
        value = self.w1(x)
        value = self.scaler(value)
        value = self.head(value)
        return value


class HyperPredictHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, init=1.0, scale=1.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scaler = Scaler(hidden_dim, init, scale)
        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.w1.weight)

    def forward(self, x):
        pred = self.w1(x)
        pred = self.scaler(pred)
        pred = self.head(pred)
        return pred
    
    
class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(input_dim + hidden_dim, 3 * hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        nn.init.orthogonal_(self.w1.weight)

    def forward(self, x, hx=None):
        if hx is None:
            hx = torch.zeros(*x.shape[:-1], self.hidden_dim, dtype=x.dtype, device=x.device)
        
        concate_o = self.w1(torch.cat([x, hx], dim=-1))
        reset, forget, cand = torch.chunk(concate_o, 3, dim=-1)
        
        reset = torch.sigmoid(reset)
        forget = torch.sigmoid(forget)
        cand = torch.tanh(cand + reset * self.w2(hx))
        
        output = forget * hx + (1.0 - forget) * cand
        return output, output
    

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim, bias=bias)
        nn.init.orthogonal_(self.w.weight)

    def forward(self, x, hx=None):
        if hx is None:
            h = torch.zeros(*x.shape[:-1], self.hidden_dim, dtype=x.dtype, device=x.device)
            c = torch.zeros(*x.shape[:-1], self.hidden_dim, dtype=x.dtype, device=x.device)
        else:
            h, c = hx
        
        concate_o = self.w(torch.cat([x, h], dim=-1))
        update, forget, output, cand = torch.chunk(concate_o, 4, dim=-1)
        
        update = torch.sigmoid(update)
        forget = torch.sigmoid(forget)
        output = torch.sigmoid(output)
        cand = torch.tanh(cand)
        
        new_c = forget * c + update * cand
        output = output * torch.tanh(new_c)
        return output, (output, new_c)
    
    
class MinRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=bias)
        nn.init.orthogonal_(self.w1.weight)

    def forward(self, x, hx=None):
        if hx is None:
            hx = torch.zeros(*x.shape[:-1], self.hidden_dim, dtype=x.dtype, device=x.device)
        
        forget = self.w1(torch.cat([x, hx], dim=-1))
        cand = torch.tanh(self.w2(x))
        
        forget = torch.sigmoid(forget)
        
        output = forget * hx + (1.0 - forget) * cand
        return output, output
