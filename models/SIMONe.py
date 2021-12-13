import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules import transformer
from einops import rearrange, repeat


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
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
        object_latent_dim=16,
        frame_latent_dim=32,
        obj_kl_loss=1e-8,
        frame_kl_loss=1e-8,
        reconstruction_loss=0.2,
        pixel_likelihood=0.8,
        n_transformer_layers=4,
        transformer_heads=4,
        transformer_dim=64,
        transformer_feedforward=1024,
    ):
        super(SIMONE, self).__init__()

        # CNN Outputs 8*8 feature map for each frame
        layers = [nn.Conv2d(input_size[1], 128, kernel_size=1, stride=2), nn.ReLU()]
        n_layers = np.log2(input_size[2] / 8)
        for i in range(int(n_layers) - 1):
            layers.append(nn.Conv2d(128, 128, kernel_size=1, stride=2))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        input_dim = 128 * 8 * 8 * input_size[0]
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
        frames = []
        for i in range(x.shape[1]):
            frames.append(self.encoder(x[:, i]))

        frames = torch.stack(frames, dim=1)
        print(frames.shape)
        frames = frames.flatten(start_dim=1).to(x.device)
        print(frames.shape)
        x = self.transformer_1(frames)
        x = self.transformer_2(x)
        return x
