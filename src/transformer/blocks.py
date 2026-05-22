import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, L, D_head]
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class ConditionEncoder(nn.Module):
    def __init__(self, input_dim=43, d_model=64, num_cond_tokens=8):
        super().__init__()
        self.num_cond_tokens = num_cond_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.cond_tokens = nn.Parameter(torch.randn(1, num_cond_tokens, d_model) * 0.02)

    def forward(self, desc):
        B = desc.size(0)
        cond = self.proj(desc)
        cond = cond.unsqueeze(1)
        cond = cond + self.cond_tokens
        return cond


class UnpatchHead(nn.Module):
    def __init__(self, d_model=128, out_channels=1, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_channels * patch_size)
        )
        
        self.smoother = nn.Conv1d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=5, 
            padding=2, 
            padding_mode='replicate'
        )

    def forward(self, x):
        x = self.head(x)
        B, N, _ = x.shape
        x = x.view(B, N * self.patch_size, self.out_channels) 
        x = x.transpose(1, 2)
        
        x = self.smoother(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        residual = q
        B, L_q, _ = q.shape
        _, L_k, _ = k.shape

        q = self.w_q(q).view(B, L_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, L_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, L_k, self.n_head, self.d_v).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  
        out = out.transpose(1, 2).contiguous().view(B, L_q, -1)

        out = self.dropout(self.fc(out))
        out = self.layer_norm(out + residual)
        return out
    



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_inner, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(d_inner, d_model, kernel_size, padding=kernel_size//2)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class CrossAttnFFTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_inner, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, x, cond):
        x = self.slf_attn(x, x, x)
        x = self.cross_attn(x, cond, cond)
        x = self.pos_ffn(x)
        return x


'''
class CrossAttnFFTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_inner, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttentionRoPE(d_model, n_head, dropout)
        self.cross_attn = MultiHeadAttentionRoPE(d_model, n_head, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, cond):
        # 1. Self-Attention (убирает шум, связывает точки графика)
        residual = x
        x = self.slf_attn(x, x, x, use_rope=True)
        x = self.norm1(x + residual)
        
        # 2. Cross-Attention (берет химию из дескрипторов)
        residual = x
        x = self.cross_attn(x, cond, cond, use_rope=False)
        x = self.norm2(x + residual)
        
        # 3. Feed Forward (сглаживание через Conv1d)
        residual = x
        x = self.pos_ffn(x)
        x = self.norm3(x + residual)
        
        return x


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE для Self-Attention
        self.inv_freq = None 

    def _apply_rope(self, q, k):
        # q, k: [B, H, L, D_head]
        L = q.shape[2]
        device = q.device
        
        if self.inv_freq is None or self.inv_freq.device != device:
            self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        
        t = torch.arange(L, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # [L, D_head/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [L, D_head]
        cos, sin = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
        
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q_rope = (q * cos) + (rotate_half(q) * sin)
        k_rope = (k * cos) + (rotate_half(k) * sin)
        return q_rope, k_rope

    def forward(self, q, k, v, use_rope=False):
        B, L_q, _ = q.shape
        B, L_k, _ = k.shape

        q = self.w_q(q).view(B, L_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, L_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, L_k, self.n_head, self.d_k).transpose(1, 2)

        if use_rope:
            q, k = self._apply_rope(q, k)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(self.dropout(attn), v).transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.fc(out)
'''