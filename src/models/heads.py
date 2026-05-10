import torch
import torch.nn as nn
import math

class SignalHead(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        hidden = out_channels // 2
        self.conv1 = nn.Conv1d(in_channels, hidden, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, hidden)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(hidden, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.SiLU()

        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x + identity   # [B, 32, L]


class DescriptorHead(nn.Module):
    """Processing descriptors for FiLM"""
    def __init__(self, in_features=43, hidden_dim=128, out_dim=64):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        self.gamma_proj = nn.Linear(hidden_dim, out_dim)
        self.beta_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, descriptors):
        # descriptors: [batch_size, 43]
        hidden = self.shared_mlp(descriptors)
        
        gamma = self.gamma_proj(hidden) # [batch_size, 32]
        beta = self.beta_proj(hidden)   # [batch_size, 32]
        
        return gamma, beta


class TimeEmbedding(nn.Module):
    """Sin embed of diffusion step t"""
    def __init__(self, base_dim, out_dim):
        super().__init__()
        self.base_dim = base_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, t):
        # t: [batch_size] 
        device = t.device
        half_dim = self.base_dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        
        # [batch_size, base_dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) 
        
        t_emb = self.mlp(embeddings) # [batch_size, out_dim]
        return t_emb