import torch
import torch.nn as nn

from src.models.heads import SignalHead, DescriptorHead, TimeEmbedding
from src.models.encoder import Encoder
from src.models.decoder import Decoder

class DiffusionSSSD(nn.Module):
    """
    Predicts added noise from noisy signal, desc and timestep.
    """
    def __init__(self, in_channels, desc_features, base_channels=32):
        super().__init__()
        
        self.signal_head = SignalHead(in_channels=in_channels, out_channels=base_channels)
        self.desc_head = DescriptorHead(in_features=desc_features, out_dim=base_channels)
        self.time_head = TimeEmbedding(base_dim=base_channels, out_dim=base_channels * 4)
        
        self.pos_emb_enc = nn.Parameter(torch.randn(1, base_channels, 968) * 0.02)  
        self.pos_emb_dec = nn.Parameter(torch.randn(1, base_channels, 968) * 0.02)
        
        self.encoder = Encoder(base_channels=base_channels, t_dim=base_channels * 4)
        self.decoder = Decoder(base_channels=base_channels, t_dim=base_channels * 4)
        
        self.out_proj = nn.Sequential(
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_channels // 2, in_channels, kernel_size=1)
        )

    def forward(self, signal, descriptors, t):
        """
        signal: [B, 2, 968] 
        descriptors: [B, 43] 
        t: [B] 
        """
        h = self.signal_head(signal)              
        
        h += self.pos_emb_enc  
        
        desc_emb = self.desc_head(descriptors) 
        t_base = self.time_head(t)                
        
        bottleneck_out, skip1 = self.encoder(h,desc_emb, t_base)
        
        h_dec = self.decoder(bottleneck_out, skip1, desc_emb, t_base)
        
        h_dec = h_dec + self.pos_emb_dec
        pred_signal = self.out_proj(h_dec) 
        
        return pred_signal