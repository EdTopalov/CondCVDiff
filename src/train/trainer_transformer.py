import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from src.utils.plots import plot_training_metrics, plot_cv_reconstruction
import torch.nn.functional as F
from src.transformer.loss import CVLoss

class CSTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, save_dir: str = "./checkpoints_transformer",
                 vol_scaler=None, cur_scaler=None, lr: float = 0.001,
                 weight_decay: float = 0.01, epochs: int = 200):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.vol_scaler = vol_scaler
        self.cur_scaler = cur_scaler
        self.epochs = epochs
        self.criterion = CVLoss().to(device)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Опыт диффузии: OneCycleLR вытягивает детали лучше, чем обычный Cosine
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,       # 30% времени на разогрев
            div_factor=25.0,
            final_div_factor=1e4
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            current = batch["current"].to(self.device)  
            features = batch["features"].to(self.device) 
            
            self.optimizer.zero_grad()
            pred = self.model(features) 

            full_loss = self.criterion(pred, current)

            full_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            self.scheduler.step() 

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

    def fit(self):
        print(f"Training NAR Transformer on {self.device}...")
        train_losses = []
        val_losses = []

        fixed_batch = next(iter(self.val_loader))
        fixed_current = fixed_batch["current"][0:1].to(self.device)  
        fixed_voltage = fixed_batch["voltage"][0:1].to(self.device)  
        fixed_features = fixed_batch["features"][0:1].to(self.device) 

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if epoch % 2 == 0 or epoch == self.epochs:
                self.model.eval()
                with torch.no_grad():
                    gen_current = self.model(fixed_features)  
                    
                gen_cur_norm = gen_current[0, 0, :].cpu().numpy()
                orig_vol_norm = fixed_voltage[0, 0, :].cpu().numpy()
                orig_cur_norm = fixed_current[0, 0, :].cpu().numpy()

                gen_cur_real = gen_cur_norm
                orig_cur_real = orig_cur_norm
                orig_vol_real = orig_vol_norm

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

            current_lr = self.optimizer.param_groups[0]['lr']
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
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(self.save_dir, "last_model.pth"))