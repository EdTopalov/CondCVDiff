import torch.nn as nn
from src.models.blocks import SSSDBlock, AttentionBlock1D  


class Encoder(nn.Module):
    def __init__(self, base_channels, t_dim):
        super().__init__()
        
        self.block1 = SSSDBlock(base_channels, cond_dim=base_channels, t_dim=t_dim)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        
        self.bottleneck1 = SSSDBlock(base_channels * 2, cond_dim=base_channels, t_dim=t_dim)
        self.attn = AttentionBlock1D(base_channels * 2)
        self.bottleneck2 = SSSDBlock(base_channels * 2, cond_dim=base_channels, t_dim=t_dim)

    def forward(self, x, desc_emb, t_emb):
        """
        x: [B, 32, 968] from SignalHead
        Outputs: bottleneck features and ONE skip-connection
        """
        x = self.block1(x, desc_emb, t_emb)
        skip1 = x 
        x = self.down1(x)
        
        x = self.bottleneck1(x, desc_emb, t_emb)
        x = self.attn(x)
        x = self.bottleneck2(x, desc_emb, t_emb)
        
        return x, skip1