import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .Attention import MultiheadAttention


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, MultiheadAttention(dim, dim_head, heads)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SIMONE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dim,
        n_transformer_layers=4,
        transformer_heads=4,
        transformer_dim=64,
        transformer_feedforward=1024,
    ):
        """
        SIMONe model
        Args:
            input_size: tuple of (B, T, C, H, W)
            hidden_dim: channels in conv layers
        """

        super(SIMONE, self).__init__()

        B, T, C, H, W = input_size

        # CNN Outputs 8*8 feature map for each frame
        layers = [
            nn.Conv2d(C, hidden_dim, kernel_size=1, stride=2),
            nn.ReLU(),
        ]
        n_layers = np.log2(input_size[-1] / 8)
        for i in range(int(n_layers) - 1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=2))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        input_dim = 8

        # input_dim = torch.prod(input_size)
        # Transformers have 4 layers, 5 heads, value=64, MLPs with hidden_layer=1024
        self.transformer_1 = Transformer(
            input_dim,
            n_transformer_layers,
            transformer_heads,
            transformer_dim,
            transformer_feedforward,
            dropout=0.1,
        )
        # Transformers have 4 layers, 5 heads, value=64, MLPs with hidden_layer=1024
        self.transformer_2 = Transformer(
            input_dim,
            n_transformer_layers,
            transformer_heads,
            transformer_dim,
            transformer_feedforward,
            dropout=0.1,
        )

    def forward(self, x):
        # encode frames:
        B, T, C, H, W = x.shape
        frames = []
        for i in range(x.shape[1]):
            frames.append(self.encoder(x[:, i]))

        frames = torch.stack(frames, dim=1)
        print(frames.shape)
        frames = frames.reshape((-1, T, frames.shape[-2], frames.shape[-1]))
        print(frames.shape)
        x = self.transformer_1(frames)
        x = self.transformer_2(x)
        return x
