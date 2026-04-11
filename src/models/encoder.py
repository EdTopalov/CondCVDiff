import torch
import torch.nn as nn
import numpy as np
from src.models.blocks import ResnetBlock1D, AttentionBlock1D
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, base_channels=64, time_dim=256):
        super().__init__()
        
        # 1st level 968
        self.down1 = ResnetBlock1D(base_channels, base_channels, cond_dim=time_dim)
        self.pool1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        
        # 2nd level 484
        self.down2 = ResnetBlock1D(base_channels * 2, base_channels * 2, cond_dim=time_dim)
        self.pool2 = nn.Conv1d(base_channels * 2, base_channels * 3, kernel_size=4, stride=2, padding=1)
        
        # 3rd level 242
        self.down3 = ResnetBlock1D(base_channels * 3, base_channels * 3, cond_dim=time_dim)
        self.pool3 = nn.Conv1d(base_channels * 3, base_channels * 4, kernel_size=4, stride=2, padding=1)
        
        # bottleneck 121
        self.mid_block1 = ResnetBlock1D(base_channels * 4, base_channels * 4, cond_dim=time_dim)
        self.mid_attn = AttentionBlock1D(base_channels * 4)
        self.mid_block2 = ResnetBlock1D(base_channels * 4, base_channels * 4, cond_dim=time_dim)

    def forward(self, x, cond):
        """
        x: [B, 32, 968] form SignalHead
        Outputs from bottleneck and skip-connections for decoder
        """
        x = self.down1(x, cond)
        skip1 = x  # [B, C, 968]
        x = self.pool1(x)
        
        x = self.down2(x, cond)
        skip2 = x  # [B, C*2, 484]
        x = self.pool2(x)
        
        x = self.down3(x, cond)
        skip3 = x  # [B, C*3, 242]
        x = self.pool3(x)
        
        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)
        
        return x, skip1, skip2, skip3