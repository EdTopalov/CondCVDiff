import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from src.utils.plots import plot_training_metrics
import torch.nn.functional as F
from src.transformer.loss import CVLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.signal import savgol_filter
from tslearn.metrics import SoftDTWLossPyTorch
from src.models.phys_loss import ElectrochemicalLoss
from src.utils.plots2 import select_diverse_samples, plot_diverse_grid

class CSTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, save_dir: str = "./checkpoints_transformer",
                 vol_scaler=None, cur_scaler=None, lr: float = 0.001,
                 weight_decay: float = 0.01, epochs: int = 200, use_smoothing: bool = False):
        
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
        self.use_smoothing = use_smoothing
        self.soft_dtw = SoftDTWLossPyTorch(gamma=0.1)
        self.electro_loss = ElectrochemicalLoss(temperature=0.01)

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
    
    def compute_dtw(self, original, reconstructed):
        """
        Dynamic Time Warping с использованием dtaidistance.
        """
        orig_f64 = original.detach().cpu().numpy().astype(np.float64).flatten()
        recon_f64 = reconstructed.detach().cpu().numpy().astype(np.float64).flatten()
        from dtaidistance import dtw
        return dtw.distance_fast(orig_f64, recon_f64, use_c=True)
    
    def _phys_gradient_loss(self, pred, target):
        pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
        target_diff = target[:, :, 1:] - target[:, :, :-1]
        
        return F.l1_loss(pred_diff, target_diff)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            current = batch["current"].to(self.device)  
            features = batch["features"].to(self.device) 
            voltage = batch["voltage"].to(self.device) 
            cycle_num = batch["cycle_num"].to(self.device)

            if self.model.training:
                features = features + torch.randn_like(features) * 0.02

            self.optimizer.zero_grad()
            
            pred, _ = self.model(features, cycle_num)

            der_loss = self._phys_gradient_loss(pred, current)
            
            phys_loss = self.electro_loss(pred, current, voltage)
            loss_total_signal = F.l1_loss(pred, current)
            full_loss = loss_total_signal + phys_loss * 0.006 + der_loss
            
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
        dtw_total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            current = batch["current"].to(self.device)
            features = batch["features"].to(self.device)
            voltage = batch["voltage"].to(self.device)
            cycle_num = batch["cycle_num"].to(self.device)

            shape_pred, _ = self.model(features, cycle_num)
            der_loss = self._phys_gradient_loss(shape_pred, current)
            phys_loss = self.electro_loss(shape_pred, current, voltage)
            loss_total_signal = F.l1_loss(shape_pred, current)
            full_loss =  loss_total_signal  + phys_loss * 0.006 + der_loss
            dtw_dist = torch.tensor(0.0)

            total_loss += full_loss.item()
            dtw_total_loss += dtw_dist
            pbar.set_postfix({"val_loss": f"{full_loss.item():.4f}"})
            
        return total_loss / len(self.val_loader),  dtw_total_loss / len(self.val_loader)

    def fit(self):
        print(f"Training NAR Transformer on {self.device}...")
        train_losses = []
        val_losses = []

        fixed_batch = next(iter(self.val_loader))
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, dtw_loss = self.val_epoch(epoch)
            self.scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if epoch % 1 == 0 or epoch == self.epochs:
                self.model.eval()
                
                sample_features = fixed_batch["features"].to(self.device)
                sample_current = fixed_batch["current"].to(self.device)
                sample_voltage = fixed_batch["voltage"].to(self.device)
                sample_cycle = fixed_batch["cycle_num"].to(self.device)
                
                with torch.no_grad():
                    shape_out, _ = self.model(sample_features, sample_cycle)
                
                gen_curr_np = shape_out.squeeze(1).cpu().numpy()
                orig_curr_np = sample_current.squeeze(1).cpu().numpy()
                orig_volt_np = sample_voltage.squeeze(1).cpu().numpy()
                
                if self.use_smoothing:
                    gen_curr_np = savgol_filter(gen_curr_np, window_length=27, polyorder=3, axis=-1)
                
                sel_gen_c, sel_orig_c, sel_orig_v = select_diverse_samples(
                    gen_curr_np, orig_curr_np, orig_volt_np, num_samples=6
                )
                
                cv_save_path = os.path.join(self.save_dir, f"grid_cv_epoch_{epoch}.png")
                plot_diverse_grid(sel_gen_c, sel_orig_c, sel_orig_v, 
                                  save_path=cv_save_path, plot_type='cv')
                
                time_save_path = os.path.join(self.save_dir, f"grid_time_epoch_{epoch}.png")
                plot_diverse_grid(sel_gen_c, sel_orig_c, sel_orig_v, 
                                  save_path=time_save_path, plot_type='time')
                
                plot_training_metrics(
                    train_losses, val_losses,
                    orig_signal=None, gen_signal=None,
                    save_path=os.path.join(self.save_dir, f"metrics_epoch_{epoch}.png")
                )
                    

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | DTW: {dtw_loss:.4f}")

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