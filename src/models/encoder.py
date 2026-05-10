import torch.nn as nn
from src.models.blocks import SSSDBlock, AttentionBlock1D  


class Encoder(nn.Module):
    def __init__(self, base_channels, t_dim):
        super().__init__()
        
        self.block1 = SSSDBlock(base_channels, cond_dim=base_channels, t_dim=t_dim)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        
        self.block2 = SSSDBlock(base_channels * 2, cond_dim=base_channels, t_dim=t_dim)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        
        self.bottleneck1 = SSSDBlock(base_channels * 4, cond_dim=base_channels, t_dim=t_dim)
        self.attn = AttentionBlock1D(base_channels * 4)
        self.bottleneck2 = SSSDBlock(base_channels * 4, cond_dim=base_channels, t_dim=t_dim)

    def forward(self, x, gamma, beta, t_emb):
        """
        x: [B, 32, 968] form SignalHead
        Outputs from bottleneck and skip-connections for decoder
        """
        x = self.block1(x, gamma, beta, t_emb)
        skip1 = x  # [B, 32, 968]
        x = self.down1(x)
        
        x = self.block2(x, gamma, beta, t_emb)
        skip2 = x  # [B, 64, 484]
        x = self.down2(x)
        
        # [B, 128, 242]
        x = self.bottleneck1(x, gamma, beta, t_emb)
        x = self.attn(x)
        x = self.bottleneck2(x, gamma, beta, t_emb)
        
        return x, skip1, skip2