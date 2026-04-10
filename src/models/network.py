import torch
import torch.nn as nn

from src.models.heads import SignalHead, DescriptorHead, TimeEmbedding
from src.models.encoder import Encoder
from src.models.decoder import Decoder

class DiffusionSSSD(nn.Module):
    """
    Predicts added noise from noisy signal, desc and timestep.
    """
    def __init__(self, in_channels=2, desc_features=43, base_channels=32):
        super().__init__()
        
        self.signal_head = SignalHead(in_channels=in_channels, out_channels=base_channels)
        self.desc_head = DescriptorHead(in_features=desc_features, out_dim=base_channels)
        self.time_head = TimeEmbedding(base_dim=128, out_dim=128) 
        
        self.encoder = Encoder(base_channels=base_channels)
        self.decoder = Decoder(base_channels=base_channels)
        
        self.out_proj = nn.Conv1d(base_channels, in_channels, kernel_size=5, padding=2)
    def forward(self, signal, descriptors, t):
        """
        signal: [B, 2, 968] signal with noise (x_t)
        descriptors: [B, 43] 
        t: [B] - diffusion step (from 0 to T)
        """
        h = self.signal_head(signal)              # [B, 32, 968]
        gamma, beta = self.desc_head(descriptors) # [B, 32], [B, 32]
        t_base = self.time_head(t)                # [B, 128]
        
        bottleneck_out, skip1, skip2 = self.encoder(h, gamma, beta, t_base)
        
        h_dec = self.decoder(bottleneck_out, skip1, skip2, gamma, beta, t_base)
        
        pred_noise = self.out_proj(h_dec)         # [B, 2, 968]
        
        return pred_noise