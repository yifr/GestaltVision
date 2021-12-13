import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SAVi(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, device):
        super(SAVi, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            PositionEmbedding(64),
            nn.LayerNorm(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
        )
