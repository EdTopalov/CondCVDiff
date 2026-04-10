import torch
import torch.nn as nn
import numpy as np
from src.models.S4 import S4Block as S4


class SSSDBlock(nn.Module):
    """Layer Norm → S4 → FiLM → SiLU → Projection + Residual"""
    def __init__(self, channels, t_dim=128, cond_dim=32, dropout=0.2):
        super().__init__()
        # GroupNorm(1, C) 
        self.norm = nn.GroupNorm(1, channels)
        
        # transposed=True -> [B, C, L]
        self.s4 = S4(d_model=channels, dropout=dropout, transposed=True)
        
        self.proj_gamma = nn.Linear(cond_dim, channels) if cond_dim != channels else nn.Identity()
        self.proj_beta = nn.Linear(cond_dim, channels) if cond_dim != channels else nn.Identity()
        
        self.proj_time = nn.Sequential(
            nn.Linear(t_dim, channels),
            nn.SiLU()
        )
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, gamma_base, beta_base, t_base):
        # x: [B, C, L]
        res = x
        
        # 1. Norm + S4
        x = self.norm(x)
        
        # (output, state)
        x, _ = self.s4(x)
        
        # 2. Preparing FiLM and Time
        # [B, 32] -> [B, C] -> [B, C, 1] 
        gamma = self.proj_gamma(gamma_base).unsqueeze(-1)
        beta = self.proj_beta(beta_base).unsqueeze(-1)
        t_emb = self.proj_time(t_base).unsqueeze(-1)
        
        # 3. Using FiLM
        x = gamma * x + beta + t_emb
        
        # 4. Activation, Dropout, Projection
        x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x + res


class Encoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        
        self.block1 = SSSDBlock(base_channels, cond_dim=base_channels)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        
        self.block2 = SSSDBlock(base_channels * 2, cond_dim=base_channels)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        
        self.bottleneck1 = SSSDBlock(base_channels * 4, cond_dim=base_channels)
        self.bottleneck2 = SSSDBlock(base_channels * 4, cond_dim=base_channels)

    def forward(self, x, gamma, beta, t_emb):
        """
        x: [B, 32, 968] form SignalHead
        Outputs from bottleneck and skip-connections for decoder
        """
        x = self.block1(x, gamma, beta, t_emb)
        skip1 = x  # [B, 32, 968]
        x = self.down1(x)
        
        x = self.block2(x, gamma, beta, t_emb)
        skip2 = x  # [B, 64, 484]
        x = self.down2(x)
        
        # [B, 128, 242]
        x = self.bottleneck1(x, gamma, beta, t_emb)
        x = self.bottleneck2(x, gamma, beta, t_emb)
        
        return x, skip1, skip2