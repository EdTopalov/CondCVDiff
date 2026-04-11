import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """ Time embeddings from transformes"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeEmbedding(nn.Module):
    def __init__(self, base_channels, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
    def forward(self, t):
        return self.mlp(t)

class SignalHead(nn.Module):
    """Input of 1D signal (voltage + current)"""
    def __init__(self, in_channels=2, out_channels=32):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        # x: [batch_size, 2, 968]
        return self.proj(x) # [batch_size, 32, 968]


class DescriptorHead(nn.Module):
    """Processing descriptors for FiLM"""
    def __init__(self, in_features=43, time_dim=256):
        super().__init__()
        self.silu_mlp = nn.Sequential(
            nn.Linear(in_features, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

    def forward(self, descriptors):
        # descriptors: [batch_size, 43]
        return self.silu_mlp(descriptors)