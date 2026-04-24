import torch
import torch.nn as nn
import math

class SignalHead(nn.Module):
    """Input of 1D signal (voltage + current)"""
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        # Point conv (kernel=1), without mixing neighbour pos
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [batch_size, 2, 968]
        return self.proj(x) # [batch_size, 32, 968]


class DescriptorHead(nn.Module):
    """Processing descriptors for FiLM"""
    def __init__(self, in_features=43, hidden_dim=128, out_dim=32):
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
    """Sin embed of diffussion step t"""
    def __init__(self, base_dim=128, out_dim=32):
        super().__init__()
        self.base_dim = base_dim
        
        # After sin
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU()
        )

    def forward(self, t):
        # t: [batch_size] - tenzor with diffussion steps  (from 0 to T)
        device = t.device
        half_dim = self.base_dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        
        # Conc of sin and cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # [batch_size, 128]
        
        t_emb = self.mlp(embeddings) # [batch_size, 32]
        return t_emb