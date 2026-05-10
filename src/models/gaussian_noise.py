import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch import Tensor

def linear_beta_schedule(timesteps: int) -> Tensor:
    """
    Linear schedule 
    Beta confirms, how much noise will be added in each step.
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

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

        betas = linear_beta_schedule(timesteps)
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

        predicted_current = self.model(signal=x_noisy, descriptors=descriptors, t=t)
        mask_pos = (x_start > 0).float()
        mask_neg = (x_start < 0).float()
        area_pos_pred = (predicted_current * mask_pos).sum(dim=-1)
        area_pos_true = (x_start * mask_pos).sum(dim=-1)
        area_neg_pred = (-predicted_current * mask_neg).sum(dim=-1)
        area_neg_true = (-x_start * mask_neg).sum(dim=-1)
        loss_area = F.mse_loss(area_pos_pred, area_pos_true) + F.mse_loss(area_neg_pred, area_neg_true)
        
        eps = 1e-6 # Чуть больше, чем 1e-8, для стабильности логарифма
        
        # Считаем площади (убедись, что они строго положительные через torch.abs)
        area_pos_pred_abs = torch.abs(area_pos_pred) + eps
        area_neg_pred_abs = torch.abs(area_neg_pred) + eps
        area_pos_true_abs = torch.abs(area_pos_true) + eps
        area_neg_true_abs = torch.abs(area_neg_true) + eps
        
        # Логарифмическое отношение
        log_ratio_pred = torch.log(area_pos_pred_abs) - torch.log(area_neg_pred_abs)
        log_ratio_true = torch.log(area_pos_true_abs) - torch.log(area_neg_true_abs)
        
        # MSE от логарифмов
        loss_ratio = F.mse_loss(log_ratio_pred, log_ratio_true)
        
        #weight = 1.0 + 0.8 * torch.abs(x_start)
        #mse_loss = (weight * (predicted_current - x_start)**2).mean()
        mse_loss = F.mse_loss(predicted_current, x_start)

        threshold = 0.5  # можно сделать настраиваемым параметром

        peak_mask_pos = (x_start > threshold).float()
        peak_mask_neg = (x_start < -threshold).float()
        peak_mask = peak_mask_pos + peak_mask_neg  # объединённая маска

        # Количество пиковых точек (для усреднения)
        n_peak_points = peak_mask.sum(dim=-1).clamp(min=1)

        # MSE только в пиковых областях (суммируем по пространству, усредняем по батчу)
        loss_peaks = ((predicted_current - x_start) ** 2 * peak_mask).sum(dim=-1) / n_peak_points
        loss_peaks = loss_peaks.mean() 
        zeros = torch.tensor(0.0, device=x_start.device)

        if epoch >= 10: 
            loss_area_delay, loss_peaks_delay, loss_ratio_delay = loss_area, loss_peaks, loss_ratio
        else:
            loss_area_delay, loss_peaks_delay, loss_ratio_delay = zeros, zeros, zeros

        weight_area  = 1e-6
        weight_ratio = 1e-5
        weight_peaks = 0.1

        return mse_loss, zeros, zeros, zeros

    @torch.no_grad()
    def p_sample(self, x, descriptors, t, t_index):
        pred_x0 = self.model(signal=x, descriptors=descriptors, t=t)

        #pred_x0.clamp_(-1.5, 1.5)

        posterior_mean_coef1 = extract(self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod), t, x.shape)
        posterior_mean_coef2 = extract((1. - self.alphas_cumprod_prev) * torch.sqrt(1. - self.betas) / (1. - self.alphas_cumprod),t, x.shape)

        model_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
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