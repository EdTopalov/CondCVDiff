import numpy as np
import pandas as pd

class LossTargetAnalyzer:
    """
    Extracts CV parameters in strict accordance with the ElectrochemicalLoss logic.
    Used for analyzing target distributions before model training.
    """
    def __init__(self, voltage_df: pd.DataFrame, current_df: pd.DataFrame):
        self.voltage = voltage_df
        self.current = current_df

    def extract_statistics(self):
        stats_list = []
        
        for i in range(len(self.voltage)):
            v = self.voltage.iloc[i].values
            c = self.current.iloc[i].values
            
            a_idx = np.argmax(c)
            c_idx = np.argmin(c)
            
            anodic_height = c[a_idx]
            anodic_pos = v[a_idx]
            
            cathodic_height = c[c_idx]
            cathodic_pos = v[c_idx]
            
            anodic_area = np.sum(c[c > 0])
            cathodic_area = np.sum(np.abs(c[c < 0]))
            
            delta_E = abs(anodic_pos - cathodic_pos)
            half_wave = (anodic_pos + cathodic_pos) / 2.0
            
            stats_list.append({
                "anodic_height": anodic_height,
                "cathodic_height": cathodic_height,
                "anodic_pos": anodic_pos,
                "cathodic_pos": cathodic_pos,
                "anodic_area": anodic_area,
                "cathodic_area": cathodic_area,
                "delta_E": delta_E,
                "half_wave_potential": half_wave
            })
            
        return pd.DataFrame(stats_list)