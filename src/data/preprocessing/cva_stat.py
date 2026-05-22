import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import find_peaks

class CVAAnalyzer:    
    def __init__(self, voltage_df: pd.DataFrame, current_df: pd.DataFrame):
        self.voltage = voltage_df
        self.current = current_df
        self.mid_idx = current_df.shape[1] // 2 

    def extract_statistics(self):
        stats_list = []
        
        for i in range(len(self.voltage)):
            v = self.voltage.iloc[i].values
            c = self.current.iloc[i].values
            
            v_anodic, c_anodic = v[:self.mid_idx], c[:self.mid_idx]
            v_cathodic, c_cathodic = v[self.mid_idx:], c[self.mid_idx:]
            
            i_pa = np.max(c_anodic)
            e_pa_idx = np.argmax(c_anodic)
            e_pa = v_anodic[e_pa_idx]
            
            i_pc = np.min(c_cathodic)
            e_pc_idx = np.argmin(c_cathodic)
            e_pc = v_cathodic[e_pc_idx]
            
            half_wave_potential = (e_pa + e_pc) / 2.0

            delta_e = abs(e_pa - e_pc)
            peak_ratio = abs(i_pa / i_pc) if i_pc != 0 else np.nan
            
            q_anodic = simpson(y=c_anodic, x=v_anodic)
            q_cathodic = simpson(y=c_cathodic, x=v_cathodic)
            q_total = q_anodic + abs(q_cathodic)
            
            # dI/dV
            def safe_max_derivative(y_arr, x_arr):
                dy = np.gradient(y_arr)
                dx = np.gradient(x_arr)
                dx_safe = np.where(dx == 0, np.nan, dx)
                deriv = dy / dx_safe
                
                if np.all(np.isnan(deriv)):
                    return 0.0
                return np.nanmax(np.abs(deriv))
            
            max_di_dv_anodic = safe_max_derivative(c_anodic, v_anodic)
            max_di_dv_cathodic = safe_max_derivative(c_cathodic, v_cathodic)

            stats_list.append({
                "I_pa": i_pa,
                "E_pa": e_pa,
                "I_pc": i_pc,
                "E_pc": e_pc,
                "Delta_E": delta_e,
                "Peak_Ratio": peak_ratio,
                "Q_anodic": q_anodic,
                "Q_cathodic": q_cathodic,
                "Q_total": q_total,
                "Max_dI_dV_anodic": max_di_dv_anodic,
                "Max_dI_dV_cathodic": max_di_dv_cathodic,
                "Half_Wave_Potential": half_wave_potential
            })
            
        return pd.DataFrame(stats_list)