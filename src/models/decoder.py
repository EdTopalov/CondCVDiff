import torch
import torch.nn as nn
from src.models.blocks import SSSDBlock


class Decoder(nn.Module):
    def __init__(self, base_channels, t_dim):
        super().__init__()

        self.up1 = nn.ConvTranspose1d(in_channels=base_channels * 2, out_channels=base_channels, kernel_size=4, stride=2, padding=1)
        self.fuse1 = nn.Conv1d(base_channels * 2, base_channels, kernel_size=1)
        
        self.block1 = SSSDBlock(base_channels, cond_dim=base_channels, t_dim=t_dim)

    def forward(self, x, skip1, desc_emb, t_emb):
        """
        x: [B, 64, 484] - output from encoder bottleneck
        skip1: [B, 32, 968] - from 1 encoder level
        """
        x = self.up1(x) 
        x = torch.cat([x, skip1], dim=1)
        x = self.fuse1(x)
        
        x = self.block1(x, desc_emb, t_emb)

        return x