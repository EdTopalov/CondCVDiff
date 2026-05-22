import os
import torch 
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.utils.plots import plot_training_metrics, plot_cv_reconstruction
from src.utils.plots2 import select_diverse_samples, plot_diverse_grid
import numpy as np
from src.data.preprocessing.pipeline import Pipeline as P

class EMA:
    def __init__(self, model, beta=0.99):
        self.beta = beta
        self.step = 0
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        self.step += 1
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.beta) * param.data + self.beta * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def setup_optimizer(model: nn.Module, lr, weight_decay, epochs):
    """
    Creates an optimizer with special rules for S4 layers.
    S4 parameters (matrices A, B, C, dt) require a small learning rate and zero weight decay.
    """
    s4_params = []
    other_params = []

    for name, p in model.named_parameters():
        if 's4' in name:
            s4_params.append(p)
        else:
            other_params.append(p)

    optimizer = optim.AdamW([
        {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': s4_params, 'lr': 0.0006, 'weight_decay': 0.0}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}: lr={group['lr']}, weight_decay={group.get('weight_decay')}, params={sum(p.numel() for p in group['params'])}")
    
    return optimizer, scheduler


class DiffusionTrainer:
    def __init__(
        self, 
        diffusion_model, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer, 
        scheduler, 
        device, 
        save_dir="./checkpoints", 
        vol_scaler=None,
        cur_scaler=None, 
        flip_the_peak=False
    ):
        self.diffusion = diffusion_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.vol_scaler = vol_scaler
        self.cur_scaler = cur_scaler
        self.ema = EMA(self.diffusion, beta=0.995)
        self.flip_the_peak = flip_the_peak
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.diffusion.train()
        total_loss = 0.0
        total_mse = 0.0
        total_area = 0.0
        total_ratio = 0.0
        total_peaks = 0.0
        
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            current = batch["current"].to(self.device)  # [B, 1, 968]
            voltage = batch["voltage"].to(self.device) 
            features = batch["features"].to(self.device) # [B, 41] (или 43)
            raw_ppm = batch["raw_ppm"].to(self.device)       
            raw_molwt = batch["raw_molwt"].to(self.device)
            features = features + torch.randn_like(features) * 0.03

            if torch.rand(1).item() < 0.15:
                features = torch.zeros_like(features)
            
            self.optimizer.zero_grad()
            
            loss_mse, loss_area, loss_ratio, loss_peaks = self.diffusion(x_start=current, descriptors=features, epoch=epoch, voltage=voltage, raw_ppm=raw_ppm, raw_molwt=raw_molwt)
            full_loss = loss_mse + loss_area + loss_ratio + loss_peaks
            
            full_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            self.ema.update(self.diffusion)
            
            total_loss += full_loss.item()
            total_mse += loss_mse.item()
            total_area += loss_area.item()
            total_ratio += loss_ratio.item()
            total_peaks += loss_peaks.item()
            
            pbar.set_postfix({"loss": f"{full_loss.item():.4f}"})
            
        return total_loss / len(self.train_loader), total_mse/len(self.train_loader), total_area/len(self.train_loader), total_ratio/len(self.train_loader), total_peaks/len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.diffusion.eval()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            current = batch["current"].to(self.device)
            features = batch["features"].to(self.device)
            voltage = batch["voltage"].to(self.device)
            raw_ppm = batch["raw_ppm"].to(self.device)      
            raw_molwt = batch["raw_molwt"].to(self.device)
            loss_mse_val, loss_area_val, loss_ratio_val, loss_peaks_val = self.diffusion(x_start=current, descriptors=features, epoch=epoch, voltage=voltage, raw_ppm=raw_ppm, raw_molwt=raw_molwt)
            sum_loss = loss_mse_val + loss_area_val + loss_ratio_val + loss_peaks_val
            
            total_loss += sum_loss.item()
            pbar.set_postfix({"val_loss": f"{sum_loss.item():.4f}"})
            
        return total_loss / len(self.val_loader)

    def fit(self, epochs):
        print(f"Teaching on {self.device}...")
        train_losses_history = []
        val_losses_history = []
        
        fixed_batch = next(iter(self.val_loader))
        for epoch in range(1, epochs + 1):
            
            train_loss, train_mse_loss, train_area_loss, train_ratio_loss, train_peaks_loss = self.train_epoch(epoch)
            self.ema.apply_shadow(self.diffusion)
            val_loss = self.val_epoch(epoch)
            
            train_losses_history.append(train_loss)
            val_losses_history.append(val_loss)

            self.scheduler.step()
            
            if epoch % 1 == 0 or epoch == epochs:
                self.diffusion.eval()
                
                num_samples = min(24, fixed_batch["current"].shape[0])
                
                sample_features = fixed_batch["features"][0:num_samples].to(self.device)
                sample_current = fixed_batch["current"][0:num_samples].to(self.device)
                sample_voltage = fixed_batch["voltage"][0:num_samples].to(self.device)
                
                with torch.no_grad():
                    gen_current = self.diffusion.sample(
                        descriptors=sample_features, 
                        shape=(num_samples, 1, sample_current.shape[-1])
                    )
                
                plots_dir = os.path.join(self.save_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                plot_training_metrics(
                    train_losses_history, 
                    val_losses_history, 
                    orig_signal=None, 
                    gen_signal=None, 
                    save_path=os.path.join(plots_dir, f"metrics_epoch_{epoch}.png")
                )

                gen_curr_np = gen_current.squeeze(1).cpu().numpy()
                orig_curr_np = sample_current.squeeze(1).cpu().numpy()
                orig_volt_np = sample_voltage.squeeze(1).cpu().numpy()
                
                if self.flip_the_peak:
                    mid_idx = gen_curr_np.shape[-1] // 2
                    gen_curr_np[:, mid_idx:] = gen_curr_np[:, mid_idx:] * -1.0
                    orig_curr_np[:, mid_idx:] = orig_curr_np[:, mid_idx:] * -1.0
                    
                sel_gen_c, sel_orig_c, sel_orig_v = select_diverse_samples(
                    gen_curr_np, orig_curr_np, orig_volt_np, num_samples=6
                )
                
                cv_save_path = os.path.join(plots_dir, f"grid_cv_epoch_{epoch}.png")
                plot_diverse_grid(sel_gen_c, sel_orig_c, sel_orig_v, 
                                  save_path=cv_save_path, plot_type='cv')
                
                time_save_path = os.path.join(plots_dir, f"grid_time_epoch_{epoch}.png")
                plot_diverse_grid(sel_gen_c, sel_orig_c, sel_orig_v, 
                                  save_path=time_save_path, plot_type='time')

            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | MSE_loss {train_mse_loss:.6f} | area_loss {train_area_loss:.6f} | ratio_loss {train_ratio_loss:.6f} | peaks_loss {train_peaks_loss:.6f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pth", epoch, val_loss)
                print(f"Saved best model (Val Loss: {val_loss:.4f})")
            self.ema.restore(self.diffusion)
        
        self.ema.apply_shadow(self.diffusion)
        self.save_checkpoint("last_model.pth", epochs, val_loss)

    def save_checkpoint(self, filename, epoch, val_loss):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.diffusion.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, path)