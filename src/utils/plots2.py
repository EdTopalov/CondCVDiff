import numpy as np
import matplotlib.pyplot as plt

def select_diverse_samples(gen_currents, orig_currents, orig_voltages, num_samples=6):
    amplitudes = np.max(gen_currents, axis=-1) - np.min(gen_currents, axis=-1)
    
    sorted_indices = np.argsort(amplitudes)
    
    if len(sorted_indices) < num_samples:
        selected_idx = sorted_indices
    else:
        indices = np.linspace(0, len(sorted_indices) - 1, num_samples).astype(int)
        selected_idx = sorted_indices[indices]
        
    return gen_currents[selected_idx], orig_currents[selected_idx], orig_voltages[selected_idx]


def plot_diverse_grid(gen_currents, orig_currents, orig_voltages, save_path, plot_type='cv'):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(len(gen_currents)):
        ax = axes[i]
        
        if plot_type == 'cv':
            x_real = orig_voltages[i]
            x_gen = orig_voltages[i]
            ax.set_xlabel("Напряжение, В", fontsize=12, fontweight='bold')
        else:
            x_real = np.arange(len(orig_currents[i]))
            x_gen = np.arange(len(gen_currents[i]))
            ax.set_xlabel("Время (шаги)", fontsize=12, fontweight='bold')
            
        ax.plot(x_real, orig_currents[i], label="Настоящий ток", color="black", linewidth=2)
        ax.plot(x_gen, gen_currents[i], label="Сгенерированный ток", color="red", linestyle="--", linewidth=2)
        
        ax.set_ylabel("Ток, А", fontsize=12, fontweight='bold')
        ax.set_title(f"Образец {i+1} (Амплитуда: {np.ptp(gen_currents[i]):.4f})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=10)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()