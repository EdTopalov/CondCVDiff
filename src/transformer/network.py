import torch
import torch.nn as nn
from src.transformer.blocks import ConditionEncoder, CrossAttnFFTBlock, UnpatchHead
import torch.nn.functional as F

class NARCVGenerator(nn.Module):
    def __init__(
        self, 
        desc_dim=41, 
        seq_len=968, 
        patch_size=8, 
        d_model=64, 
        n_head=4, 
        d_inner=32, 
        num_layers=4, 
        num_cond_tokens=4,
        dropout=0.2,  
        use_cycle_feat=True
    ):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.use_cycle_feat = use_cycle_feat
        self.cond_encoder = ConditionEncoder(
            input_dim=desc_dim, d_model=d_model, num_cond_tokens=num_cond_tokens
        )

        if self.use_cycle_feat:
            self.cycle_embedder = nn.Sequential(
                nn.Linear(1, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, d_model)
            )
        
        self.positional_queries = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            CrossAttnFFTBlock(d_model, n_head, d_inner, dropout) 
            for _ in range(num_layers)
        ])
        
        self.unpatch_head = UnpatchHead(d_model=d_model, out_channels=1, patch_size=patch_size)

        self.smoother = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode='replicate')

        self.calibrator = nn.Sequential(
            nn.Linear(desc_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2) 
        )

    def forward(self, descriptors, cycle_num):
        B = descriptors.size(0)
        
        cond_tokens = self.cond_encoder(descriptors)

        if self.use_cycle_feat and cycle_num is not None:
            cycle_token = self.cycle_embedder(cycle_num).unsqueeze(1)
            
            cond_tokens = torch.cat([cond_tokens, cycle_token], dim=1)

        x = self.positional_queries.expand(B, -1, -1)
        #x = F.dropout(x, p=0.1, training=self.training)
        
        for layer in self.layers:
            x = layer(x, cond_tokens)
            
        shape_signal = self.unpatch_head(x)
        shape_signal = self.smoother(shape_signal) # [B, 1, 968]
        
        _ = 0.0

        return shape_signal, _