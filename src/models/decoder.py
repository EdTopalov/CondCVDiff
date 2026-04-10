from torch import nn 
import torch
import torch.nn as nn

from src.models.encoder import SSSDBlock


class Decoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()

        self.up2 = nn.ConvTranspose1d(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1)
        self.block2 = SSSDBlock(base_channels * 2, cond_dim=base_channels)

        # Level 1
        # Input: 64 channels | Output: 32 channels
        self.up1 = nn.ConvTranspose1d(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.block1 = SSSDBlock(base_channels, cond_dim=base_channels)

    def forward(self, x, skip1, skip2, gamma, beta, t_emb):
        """
        x: [B, 128, 242] - output from encoder
        skip2: [B, 64, 484] - from 2 encoder level (saved signal)
        skip1: [B, 32, 968] - from 1 encoder level (saved signal)
        """
        
        # Level 2
        x = self.up2(x) # [B, 128, 242] -> [B, 64, 484]
        x = x + skip2
        x = self.block2(x, gamma, beta, t_emb)

        # Level 1 
        x = self.up1(x) # [B, 64, 484] -> [B, 32, 968]
        x = x + skip1
        x = self.block1(x, gamma, beta, t_emb)

        return x

