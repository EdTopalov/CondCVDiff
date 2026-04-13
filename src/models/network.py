import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads import SignalHead, DescriptorHead, TimeEmbedding
from src.models.encoder import Encoder
from src.models.decoder import Decoder

class DiffusionAttn(nn.Module):
    """
    Predicts added noise from noisy signal, desc and timestep.
    """
    def __init__(self, in_channels=1, desc_features=43, base_channels=64):
        super().__init__()
        
        time_dim = base_channels * 4

        self.signal_head = SignalHead(in_channels=in_channels, out_channels=base_channels)
        self.desc_head = DescriptorHead(in_features=desc_features, time_dim=time_dim)
        self.time_head = TimeEmbedding(base_channels=base_channels, time_dim=time_dim) 
        
        self.pos_emb = nn.Parameter(torch.randn(1, base_channels, 968) * 0.02)  # Learnable positional embeddings

        self.encoder = Encoder(base_channels=base_channels, time_dim=time_dim)
        self.decoder = Decoder(base_channels=base_channels, time_dim=time_dim)
        
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_proj = nn.Conv1d(base_channels, in_channels, kernel_size=5, padding=2)

    def forward(self, signal, descriptors, t):
        """
        signal: [B, 1, 968] signal with noise (x_t)
        descriptors: [B, 43] 
        t: [B] - diffusion step (from 0 to T)
        """
        h = self.signal_head(signal)     

        h = h + self.pos_emb

        desc_emb = self.desc_head(descriptors)    
        t_emb = self.time_head(t)                 
        
        cond = t_emb + desc_emb                   
        
        bottleneck_out, skip1, skip2, skip3 = self.encoder(h, cond)
        
        h_dec = self.decoder(bottleneck_out, skip1, skip2, skip3, cond)
        
        h_dec = self.out_norm(h_dec)
        h_dec = F.silu(h_dec)
        pred_noise = self.out_proj(h_dec)         
        
        return pred_noise