import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from src.utils.plots import plot_training_metrics, plot_cv_reconstruction
import torch.nn.functional as F

class CSTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, save_dir: str = "./checkpoints_transformer",
                 vol_scaler=None, cur_scaler=None, lr: float = 0.001,
                 weight_decay: float = 0.01, epochs: int = 200,
                 area_loss_start_epoch: int = 10, area_loss_weight: float = 0.001,
                 use_weight_mask: bool = True, peak_weight: float = 5.0,
                 peak_ranges: list = None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.vol_scaler = vol_scaler
        self.cur_scaler = cur_scaler
        self.area_loss_start_epoch = area_loss_start_epoch
        self.area_loss_weight = area_loss_weight

        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.use_weight_mask = use_weight_mask
        if use_weight_mask:
            if peak_ranges is None:
                peak_ranges = [(0, 200), (450, 650)]
            weight = torch.ones(1, 1, 968)
            for start, end in peak_ranges:
                weight[:, :, start:end] = peak_weight
            self.register_buffer('weight_mask', weight)
        else:
            self.weight_mask = None

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.to(self.device))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            current = batch["current"].to(self.device)  
            features = batch["features"].to(self.device) 
            
            self.optimizer.zero_grad()
            pred = self.model(features) 

            # 1. Основной MSE
            if self.use_weight_mask:
                diff = pred - current
                mse_loss = (self.weight_mask * diff ** 2).mean()
            else:
                mse_loss = F.mse_loss(pred, current)
            
            # 2. Границы
            bounds = F.relu(torch.abs(pred) - 1.0) ** 2
            bounds_loss = bounds.mean()

            pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
            true_grad = current[:, :, 1:] - current[:, :, :-1]
            grad_loss = F.mse_loss(pred_grad, true_grad)
            
            # 4. Area loss
            area_loss = torch.tensor(0.0, device=self.device)
            if epoch >= self.area_loss_start_epoch:
                mask_pos = (current > 0).float()
                mask_neg = (current < 0).float()
                pos_pred = (pred * mask_pos).sum(dim=-1)
                pos_true = (current * mask_pos).sum(dim=-1)
                neg_pred = (-pred * mask_neg).sum(dim=-1)
                neg_true = (-current * mask_neg).sum(dim=-1)
                area_loss = F.mse_loss(pos_pred, pos_true) + F.mse_loss(neg_pred, neg_true)
            full_loss = mse_loss + 0.01 * bounds_loss + 0.1 * grad_loss + self.area_loss_weight * area_loss
            
            full_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += full_loss.item()
            pbar.set_postfix({"loss": f"{full_loss.item():.4f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            current = batch["current"].to(self.device)
            features = batch["features"].to(self.device)
            pred = self.model(features)
            
            mse_loss = F.mse_loss(pred, current)
            total_loss += mse_loss.item()
            pbar.set_postfix({"val_loss": f"{mse_loss.item():.4f}"})
            
        return total_loss / len(self.val_loader)


    def fit(self, epochs):
        print(f"Training Transformer on {self.device}...")
        train_losses = []
        val_losses = []

        fixed_batch = next(iter(self.val_loader))
        fixed_current = fixed_batch["current"][0:1].to(self.device)  # [1,1,968]
        fixed_voltage = fixed_batch["voltage"][0:1].to(self.device)  # [1,1,968]
        fixed_features = fixed_batch["features"][0:1].to(self.device) # [1,43]

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step()

            if epoch % 5 == 0 or epoch == epochs:
                self.model.eval()
                with torch.no_grad():
                    gen_current = self.model(fixed_features)  # [1,1,968]
                gen_cur_norm = gen_current[0, 0, :].cpu().numpy()
                orig_vol_norm = fixed_voltage[0, 0, :].cpu().numpy()
                orig_cur_norm = fixed_current[0, 0, :].cpu().numpy()

                gen_cur_real = self.cur_scaler.inverse_transform(gen_cur_norm.reshape(-1, 1)).flatten()
                orig_cur_real = self.cur_scaler.inverse_transform(orig_cur_norm.reshape(-1, 1)).flatten()
                orig_vol_real = self.vol_scaler.inverse_transform(orig_vol_norm.reshape(-1, 1)).flatten()

                gen_signal_real = np.stack([orig_vol_real, gen_cur_real])
                orig_signal_real = np.stack([orig_vol_real, orig_cur_real])

                plot_training_metrics(
                    train_losses, val_losses,
                    orig_signal=orig_signal_real,
                    gen_signal=gen_signal_real,
                    save_path=os.path.join(self.save_dir, f"metrics_epoch_{epoch}.png")
                )
                plot_cv_reconstruction(
                    orig_signal=orig_signal_real,
                    gen_signal=gen_signal_real,
                    save_path=os.path.join(self.save_dir, f"cv_curve_epoch_{epoch}.png")
                )

            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(self.save_dir, "best_model.pth"))
                print(f"Saved best model (Val Loss: {val_loss:.4f})")

        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(self.save_dir, "last_model.pth"))