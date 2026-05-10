from torch import nn 
import torch
import torch.nn as nn

from src.models.blocks import SSSDBlock, AttentionBlock1D


class Decoder(nn.Module):
    def __init__(self, base_channels, t_dim):
        super().__init__()

        self.up2 = nn.ConvTranspose1d(in_channels=base_channels * 4, out_channels=base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.fuse2 = nn.Conv1d(base_channels * 4, base_channels * 2, kernel_size=1)
        self.block2 = SSSDBlock(base_channels * 2, cond_dim=base_channels, t_dim=t_dim)

        #self.attn = AttentionBlock1D(base_channels * 2)

        # Level 1
        # Input: 64 channels | Output: 32 channels
        self.up1 = nn.ConvTranspose1d(in_channels=base_channels * 2, out_channels=base_channels, kernel_size=4, stride=2, padding=1)
        self.fuse1 = nn.Conv1d(base_channels * 2, base_channels, kernel_size=1)
        self.block1 = SSSDBlock(base_channels, cond_dim=base_channels, t_dim=t_dim)
        
        self.block1_extra = SSSDBlock(base_channels, cond_dim=base_channels, t_dim=t_dim)

    def forward(self, x, skip1, skip2, gamma, beta, t_emb):
        """
        x: [B, 128, 242] - output from encoder
        skip2: [B, 64, 484] - from 2 encoder level (saved signal)
        skip1: [B, 32, 968] - from 1 encoder level (saved signal)
        """
        
        # Level 2
        x = self.up2(x) # [B, 128, 242] -> [B, 64, 484]
        x = torch.cat([x, skip2], dim=1)
        x = self.fuse2(x)
        x = self.block2(x, gamma, beta, t_emb)


        # Level 1 
        x = self.up1(x) # [B, 64, 484] -> [B, 32, 968]
        x = torch.cat([x, skip1], dim=1)
        x = self.fuse1(x)
        x = self.block1(x, gamma, beta, t_emb)

        x = self.block1_extra(x, gamma, beta, t_emb)

        return x

