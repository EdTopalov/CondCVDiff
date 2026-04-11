from torch import nn 
import torch
import torch.nn as nn

from src.models.blocks import ResnetBlock1D, AttentionBlock1D


class Decoder(nn.Module):
    def __init__(self, base_channels=64, time_dim=256):
        super().__init__()
        
        self.up_pool3 = nn.ConvTranspose1d(base_channels * 4, base_channels * 3, kernel_size=4, stride=2, padding=1)
        self.up3 = ResnetBlock1D(base_channels * 6, base_channels * 3, cond_dim=time_dim) # 3 from pooling + 3 from skip 
        
        # 242 -> 484
        self.up_pool2 = nn.ConvTranspose1d(base_channels * 3, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = ResnetBlock1D(base_channels * 4, base_channels * 2, cond_dim=time_dim) # 2+2
        
        #484 -> 968
        self.up_pool1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = ResnetBlock1D(base_channels * 2, base_channels, cond_dim=time_dim) # 1+1


    def forward(self, x, skip1, skip2, skip3, cond):
        
        x = self.up_pool3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up3(x, cond)
        
        x = self.up_pool2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x, cond)
        
        x = self.up_pool1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x, cond)
        
        return x

