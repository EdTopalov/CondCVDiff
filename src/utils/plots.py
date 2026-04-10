import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

def plot_training_metrics(train_losses, val_losses, orig_signal, gen_signal, save_path=None):

    orig_signal = _to_numpy(orig_signal)
    gen_signal = _to_numpy(gen_signal)
    
    orig_vol, orig_cur = orig_signal[0], orig_signal[1]
    gen_vol, gen_cur = gen_signal[0], gen_signal[1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    if val_losses:
        ax1.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    ax1.set_title('Функция потерь (MSE)')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    time_steps = np.arange(len(orig_vol))
    
    ax2.plot(time_steps, orig_vol, label='Orig Voltage', color='navy', alpha=0.8)
    ax2.plot(time_steps, orig_cur, label='Orig Current', color='darkred', alpha=0.8)
    
    ax2.plot(time_steps, gen_vol, label='Gen Voltage', color='cyan', linestyle='--', linewidth=2)
    ax2.plot(time_steps, gen_cur, label='Gen Current', color='orange', linestyle='--', linewidth=2)
    
    ax2.set_title('Сравнение 1D каналов (Развертка по времени)')
    ax2.set_xlabel('Индекс (0 - 968)')
    ax2.set_ylabel('Амплитуда')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_cv_reconstruction(orig_signal, gen_signal, save_path=None):
    orig_signal = _to_numpy(orig_signal)
    gen_signal = _to_numpy(gen_signal)
    
    orig_vol, orig_cur = orig_signal[0], orig_signal[1]
    gen_vol, gen_cur = gen_signal[0], gen_signal[1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(gen_vol, gen_cur, color='red', linewidth=2)
    ax1.set_title('Сгенерированная Вольтамперограмма', fontsize=12)
    ax1.set_xlabel('Напряжение (V)')
    ax1.set_ylabel('Ток (I)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(orig_vol, orig_cur, color='blue', linewidth=2)
    ax2.set_title('Оригинальная Вольтамперограмма (Ground Truth)', fontsize=12)
    ax2.set_xlabel('Напряжение (V)')
    ax2.set_ylabel('Ток (I)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    x_min = min(orig_vol.min(), gen_vol.min())
    x_max = max(orig_vol.max(), gen_vol.max())
    y_min = min(orig_cur.min(), gen_cur.min())
    y_max = max(orig_cur.max(), gen_cur.max())
    
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    
    ax1.set_xlim([x_min - margin_x, x_max + margin_x])
    ax1.set_ylim([y_min - margin_y, y_max + margin_y])
    ax2.set_xlim([x_min - margin_x, x_max + margin_x])
    ax2.set_ylim([y_min - margin_y, y_max + margin_y])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()