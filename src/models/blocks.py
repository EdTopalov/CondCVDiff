import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock1D(nn.Module):
    """ResNet with FiLM for conditioning"""
    def __init__(self, in_channels, out_channels, cond_dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2)
        )
        
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)

class AttentionBlock1D(nn.Module):
    """Self-Attention"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, C, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        attn = torch.einsum("bcl,bcs->bls", q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum("bls,bcs->bcl", attn, v)
        return x + self.proj(out)