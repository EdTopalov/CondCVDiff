import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch import Tensor

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Softly adds noise so the signal structure is preserved longer.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    """
    Helper function. Extracts the coefficients a at indices t
    and reshapes them for correct broadcasting with tensor x.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        #w = torch.ones(1, 1, 968)
        #w[:, :, 0:200] = 5.0
        #w[:, :, 450:650] = 5.0
        #self.register_buffer('weight_mask', w)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor = None) -> Tensor:
        """
        Forward process. Adds noise to x_start up to step t.
        Used during training.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x_start: Tensor, descriptors: Tensor, epoch) -> Tensor:
        """
        This is called when model(signal, descriptors) is invoked in the training loop.
        """
        b, c, l = x_start.shape
        device = x_start.device

        t = torch.randint(0, self.timesteps, (b,), device=device).long()

        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    
        predicted_signal = self.model(signal=x_noisy, descriptors=descriptors, t=t)
        #---------------------------------------------------------
        l1_err = F.l1_loss(predicted_signal, x_start, reduction="none")
        mse_err = F.mse_loss(predicted_signal, x_start, reduction="none")
        hybrid_err = l1_err + 2.0 * mse_err
        peak_weight = 1.0 + 2.0 * torch.abs(x_start)
        loss_main = hybrid_err * peak_weight
        loss_base = loss_main.mean()
        #---------------------------------------------------------
        mse_err_t = F.mse_loss(predicted_signal, x_start)

        # Area loss
        L = x_start.shape[-1]
        mid_idx = L // 2
        eps = 1e-8

        pred_relu = F.relu(predicted_signal)
        true_relu = F.relu(x_start)

        area_pred_1 = torch.sum(pred_relu[:, :, :mid_idx], dim=-1)
        area_true_1 = torch.sum(true_relu[:, :, :mid_idx], dim=-1)
        
        area_pred_2 = torch.sum(pred_relu[:, :, mid_idx:], dim=-1)
        area_true_2 = torch.sum(true_relu[:, :, mid_idx:], dim=-1)

        loss_area = F.mse_loss(area_pred_1, area_true_1) + F.mse_loss(area_pred_2, area_true_2)
        
        # Peaks loss
        indices = torch.linspace(0.0, 1.0, L, device=x_start.device).view(1, 1, -1)

        com_pred_1 = torch.sum(pred_relu[:, :, :mid_idx] * indices[:, :, :mid_idx], dim=-1) / (area_pred_1 + eps)
        com_true_1 = torch.sum(true_relu[:, :, :mid_idx] * indices[:, :, :mid_idx], dim=-1) / (area_true_1 + eps)
        
        com_pred_2 = torch.sum(pred_relu[:, :, mid_idx:] * indices[:, :, mid_idx:], dim=-1) / (area_pred_2 + eps)
        com_true_2 = torch.sum(true_relu[:, :, mid_idx:] * indices[:, :, mid_idx:], dim=-1) / (area_true_2 + eps)
        
        loss_pos = F.l1_loss(com_pred_1, com_true_1) + F.l1_loss(com_pred_2, com_true_2)

        '''
        L = x_start.shape[-1]
        tail_mask = (torch.arange(L, device=x_start.device) >= 500).float()
        tail_mask = tail_mask.view(1, 1, -1)
        pos_mask = (x_start >= 0).float()
        neg_mask = (x_start < 0).float()
        
        if epoch >= 150:
            wt = 2.0
        else:
            wt = 0.0

        loss_tail = mse_err * tail_mask * wt

        loss_pos = (l1_err + 3.0 * mse_err) * pos_mask #4 lr 0.0006
        neg_weight = 1.0 + 3.5 * torch.abs(x_start) #5
        loss_neg = (l1_err * neg_weight) * neg_mask

        loss_base = (loss_pos + loss_neg).mean()'''

        # ---------------------------------------------------------------------- #        
        '''
        shift = 0.0 

        l1_err = F.l1_loss(predicted_signal, x_start, reduction="none")
        mse_err = F.mse_loss(predicted_signal, x_start, reduction="none")

        hybrid_err = mse_err + 0.4 * l1_err
        dev = torch.abs(x_start - shift)
        peak_weight = 1.0 + 3.0 * dev

        loss_main = (hybrid_err * peak_weight).mean()
        '''

        '''
        diff1_pred = predicted_signal[:, :, 1:] - predicted_signal[:, :, :-1]
        diff1_true = x_start[:, :, 1:] - x_start[:, :, :-1]
        loss_diff1 = F.l1_loss(diff1_pred, diff1_true)

        diff2_pred = diff1_pred[:, :, 1:] - diff1_pred[:, :, :-1]
        diff2_true = diff1_true[:, :, 1:] - diff1_true[:, :, :-1]
        loss_diff2 = F.l1_loss(diff2_pred, diff2_true)

        total_loss = loss_base + 1.0 * loss_diff1 + 1.0 * loss_diff2
        '''
        zeros = torch.tensor(0.0, device=device)
        if epoch >= 15:
            act = 1.0
            act2 = 1.5
        else:
            act2 = 1
            act = 0

        #return mse_err_t * act2, loss_area * 0.00001 * act, loss_pos * 0.5 * act, zeros
        return mse_err_t, zeros, zeros, zeros 
    @torch.no_grad()
    def p_sample(self, x, descriptors, t, t_index):
        pred_x0 = self.model(signal=x, descriptors=descriptors, t=t)
        
        shift = 0.0 #new
        
        pred_centered = pred_x0 - shift #new
        
        s = torch.amax(torch.abs(pred_centered), dim=(1, 2), keepdim=True)
        limit = torch.tensor(1.0, device=pred_x0.device)
        s_scale = torch.maximum(s, limit)

        pred_centered = pred_centered * (limit / s_scale)
        pred_x0 = pred_centered + shift #new

        posterior_mean_coef1 = extract(self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod), t, x.shape)
        posterior_mean_coef2 = extract((1. - self.alphas_cumprod_prev) * torch.sqrt(1. - self.betas) / (1. - self.alphas_cumprod),t, x.shape)

        model_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            temperature = 1.1
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, descriptors, shape):
        """
        Full generation from scratch (Inference).
        shape: (batch_size, 2, 968)
        """
        device = next(self.model.parameters()).device
        b = shape[0]
        
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, descriptors, t, i)
            
        return img