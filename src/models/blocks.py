import torch.nn as nn
import torch 
from src.models.S4 import S4Block as S4
import torch.nn.functional as F


class AttentionBlock1D(nn.Module):
    """Self-Attention with QKV"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels) 
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, C, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # Scaled dot-product attention
        attn = torch.einsum("bcl,bcs->bls", q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.einsum("bls,bcs->bcl", attn, v)
        out = self.proj(out)
        out = self.drop(out)
        return x + out

class SSSDBlock(nn.Module):
    """Layer Norm → S4 → FiLM → SiLU → Projection + Residual"""
    def __init__(self, channels, t_dim, cond_dim, dropout=0.3):
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
        self.dropout = nn.Dropout1d(dropout) 
        
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
