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
        attn = torch.einsum("bcl,bcs->bls", q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.einsum("bls,bcs->bcl", attn, v)
        out = self.proj(out)
        out = self.drop(out)
        return x + out

class SSSDBlock(nn.Module):
    """Layer Norm → S4 → Conc → SiLU → Projection + Residual"""
    def __init__(self, channels, t_dim, cond_dim, dropout=0.3):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        
        self.s4 = S4(d_model=channels, dropout=dropout, transposed=True)
        
        self.mix_conv = nn.Conv1d(channels + cond_dim, channels, kernel_size=1)
        self.proj_time = nn.Sequential(
            nn.Linear(t_dim, channels),
            nn.SiLU()
        )
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout1d(dropout) 
        
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, desc_emb, t_base):
        res = x
        
        x = self.norm(x)
        
        x, _ = self.s4(x)
        
        desc_expanded = desc_emb.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x_cat = torch.cat([x, desc_expanded], dim=1) 

        t_emb = self.proj_time(t_base).unsqueeze(-1)
        
        x = self.mix_conv(x_cat) + t_emb
        
        x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x + res
