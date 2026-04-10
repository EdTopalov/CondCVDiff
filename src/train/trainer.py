import os
import torch 
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.utils.plots import plot_training_metrics, plot_cv_reconstruction
import numpy as np
from src.data.preprocessing.pipeline import Pipeline as P


def setup_optimizer(model: nn.Module, lr=0.001, weight_decay=0.01, epochs=100):
    """
    Создает оптимизатор с особыми правилами для слоев S4.
    Параметры S4 (матрицы A, B, C, dt) требуют маленького LR и нулевого Weight Decay.
    """
    all_parameters = list(model.parameters())

    general_params = [p for p in all_parameters if not hasattr(p, "_optim")]
    
    optimizer = optim.AdamW(general_params, lr=lr, weight_decay=weight_decay)

    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
        cur_scaler=None
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
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.diffusion.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            signal = batch["signal_1d"].to(self.device)  # [B, 2, 968]
            features = batch["features"].to(self.device) # [B, 41] (или 43)
            
            self.optimizer.zero_grad()
            
            loss = self.diffusion(x_start=signal, descriptors=features)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.diffusion.eval()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            signal = batch["signal_1d"].to(self.device)
            features = batch["features"].to(self.device)
            
            loss = self.diffusion(x_start=signal, descriptors=features)
            
            total_loss += loss.item()
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
            
        return total_loss / len(self.val_loader)

    def fit(self, epochs):
        print(f"Teaching on {self.device}...")
        train_losses_history = []
        val_losses_history = []
        
        fixed_batch = next(iter(self.val_loader))
        fixed_signal = fixed_batch["signal_1d"][0:1].to(self.device) # 1st element of the batch, shape: [1, 2, 968]
        fixed_features = fixed_batch["features"][0:1].to(self.device)
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            
            train_losses_history.append(train_loss)
            val_losses_history.append(val_loss)

            self.scheduler.step()
            
            if epoch % 10 == 0 or epoch == epochs:
                self.diffusion.eval()
                with torch.no_grad():
                    # generates from noise shape=(1, 2, 968)
                    gen_signal = self.diffusion.sample(
                        descriptors=fixed_features, 
                        shape=(1, 2, fixed_signal.shape[-1])
                    )
                
                plots_dir = os.path.join(self.save_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                gen_vol_norm = gen_signal[0, 0, :].cpu().numpy()
                gen_cur_norm = gen_signal[0, 1, :].cpu().numpy()
                orig_vol_norm = fixed_signal[0, 0, :].cpu().numpy()
                orig_cur_norm = fixed_signal[0, 1, :].cpu().numpy()


                gen_vol_real = self.vol_scaler.inverse_transform(gen_vol_norm.reshape(-1, 1)).flatten()
                gen_cur_real = self.cur_scaler.inverse_transform(gen_cur_norm.reshape(-1, 1)).flatten()

                orig_vol_real = self.vol_scaler.inverse_transform(orig_vol_norm.reshape(-1, 1)).flatten()
                orig_cur_real = self.cur_scaler.inverse_transform(orig_cur_norm.reshape(-1, 1)).flatten()
                
                gen_signal_real = np.stack([gen_vol_real, gen_cur_real])
                orig_signal_real = np.stack([orig_vol_real, orig_cur_real])

                plot_training_metrics(
                    train_losses_history, 
                    val_losses_history, 
                    orig_signal=orig_signal_real, 
                    gen_signal=gen_signal_real, 
                    save_path=os.path.join(plots_dir, f"metrics_epoch_{epoch}.png")
                )
                
                plot_cv_reconstruction(
                    orig_signal=orig_signal_real, 
                    gen_signal=gen_signal_real, 
                    save_path=os.path.join(plots_dir, f"cv_curve_epoch_{epoch}.png")
                )

            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pth", epoch, val_loss)
                print(f"Saved best model (Val Loss: {val_loss:.4f})")
                
        self.save_checkpoint("last_model.pth", epochs, val_loss)

    def save_checkpoint(self, filename, epoch, val_loss):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.diffusion.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, path)