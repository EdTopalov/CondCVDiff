import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer.blocks import ConditionEncoder, CrossAttnFFTBlock, UnpatchHead


class CurveEncoder(nn.Module):
    def __init__(self, seq_len=968, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

    def forward(self, x):
        features = self.net(x)
        return self.fc_mu(features), self.fc_logvar(features)


class NARCVGeneratorCVAE(nn.Module):
    def __init__(self, desc_dim=43, seq_len=968, patch_size=8, d_model=64, n_head=4, d_inner=64, num_layers=4, num_cond_tokens=8, latent_dim=16, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.curve_encoder = CurveEncoder(seq_len=seq_len, latent_dim=latent_dim)
        
        self.z_projection = nn.Linear(latent_dim, d_model)
        
        self.num_patches = seq_len // patch_size
        self.cond_encoder = ConditionEncoder(desc_dim, d_model, num_cond_tokens)
        self.positional_queries = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.layers = nn.ModuleList([CrossAttnFFTBlock(d_model, n_head, d_inner, dropout) for _ in range(num_layers)])
        self.unpatch_head = UnpatchHead(d_model, out_channels=1, patch_size=patch_size)
        self.smoother = nn.Conv1d(1, 1, kernel_size=5, padding=2, padding_mode='replicate')

        self.scalar_head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.SiLU(),
            nn.Linear(16, 2)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, descriptors, current_gt=None):
        B = descriptors.size(0)
        
        if current_gt is not None:
            mu, logvar = self.curve_encoder(current_gt)
            z = self.reparameterize(mu, logvar)
        else:
            z = torch.randn(B, self.latent_dim, device=descriptors.device)
            mu, logvar = None, None 
            
        z_emb = self.z_projection(z).unsqueeze(1) # [B, 1, d_model]
        cond_tokens = self.cond_encoder(descriptors) # [B, num_cond, d_model]
        
        cond_tokens = cond_tokens + z_emb 
        
        x = self.positional_queries.expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, cond_tokens)
            
        shape_signal = self.unpatch_head(x)
        shape_signal = self.smoother(shape_signal)
        
        cond_pooled = cond_tokens.mean(dim=1) 
        scalars = self.scalar_head(cond_pooled)
        log_range = scalars[:, 0:1]
        min_val = scalars[:, 1:2]
        
        return shape_signal, min_val, log_range, mu, logvar