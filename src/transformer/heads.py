import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, d_model=64, patch_size=4, max_seq_len=968):
        super().__init__()
        self.patch_size = patch_size
        
        self.proj = nn.Conv1d(
            in_channels, 
            d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        num_patches = max_seq_len // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.proj(x)                  
        x = x.transpose(1, 2)             
        
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.norm(x)
        return x

class UnpatchHead(nn.Module):
    def __init__(self, d_model=128, out_channels=1, patch_size=10):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_channels * patch_size)
        )

    def forward(self, x):
        x = self.head(x)                  
        B, N, _ = x.shape
        x = x.view(B, N * self.patch_size, self.out_channels) 
        x = x.transpose(1, 2)                                 
        return x