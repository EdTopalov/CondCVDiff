import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch

def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

def plot_training_metrics(train_losses, val_losses, orig_signal=None, gen_signal=None, save_path=None):
    if orig_signal is None or gen_signal is None:
        fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
        
        ax1.plot(train_losses, label='Тренировочная ошибка', color='#1F77B4', linewidth=1.5)
        if val_losses:
            ax1.plot(val_losses, label='Валидационная ошибка', color='#D62728', linewidth=1.5)
            
        ax1.set_title('Функция потерь', fontsize=14, fontweight='bold', pad=12)
        ax1.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Ошибка', fontsize=12, fontweight='bold')
        
        ax1.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax1.legend(fontsize=11, frameon=True, edgecolor='black')
        
        for spine in ax1.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return 

    orig_signal = _to_numpy(orig_signal)
    gen_signal = _to_numpy(gen_signal)
    
    orig_cur = orig_signal[1]
    gen_cur = gen_signal[1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    
    ax1.plot(train_losses, label='Тренировочная ошибка', color='#1F77B4', linewidth=1.5)
    if val_losses:
        ax1.plot(val_losses, label='Валидационная ошибка', color='#D62728', linewidth=1.5)
        
    ax1.set_title('Функция потерь', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ошибка', fontsize=12, fontweight='bold')
    
    ax1.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax1.legend(fontsize=11, frameon=True, edgecolor='black')
    
    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
        
    time_steps = np.arange(len(orig_cur))
    
    ax2.plot(time_steps, orig_cur, label='Настоящий ток', color='black', linewidth=1.5)
    ax2.plot(time_steps, gen_cur, label='Сгенерированный ток', color='#D62728', linestyle='--', linewidth=1.5)
    
    ax2.set_title('Развертка тока по времени', fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel('Время (индекс)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Амплитуда тока', fontsize=12, fontweight='bold')
    
    ax2.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax2.legend(fontsize=11, frameon=True, edgecolor='black')
    
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
        
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