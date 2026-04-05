import torch
from torch.utils.data import Dataset
import numpy as np

descriptors_name = ['MolWt', 'MolLogP', 'NumRotatableBonds', 'TPSA',
       'FractionCSP3', 'NumAromaticRings', 'Chi0', 'Chi1', 'Kappa1', 'Kappa2',
       'BertzCT', 'BalabanJ', 'PEOE_VSA1', 'PEOE_VSA2', 'SMR_VSA1', 'SMR_VSA2',
       'EState_VSA1', 'EState_VSA2', 'MaxEStateIndex', 'MinEStateIndex',
       'FpDensityMorgan1', 'SMR_VSA10', 'VSA_EState2', 'VSA_EState3',
       'BCUT2D_MWHI', 'BCUT2D_LOGP', 'BCUT2D_CHGHI', 'Nitrogen', 'Sulfur',
       'AmineGroup', 'G Eh', 'HOMO eV', 'LUMO eV', 'μ D',
       'Final entropy term Eh', 'Total enthalpy Eh', 'Electronic entropy Eh',
       'Vibrational entropy Eh', 'Rotational entropy Eh',
       'Translational entropy Eh', 'ppm']

class CVADataset(Dataset):
    def __init__(self, vol, cur, desc_df):
        

        v_df = vol.reset_index(drop=True)
        c_df = cur.reset_index(drop=True)
        d_df = desc_df.reset_index(drop=True)

        cols_to_drop = ["Inhibitor", "Num_of_Cycle", "ppm"]
        v_clean = v_df.drop(columns=[c for c in cols_to_drop if c in v_df.columns])
        c_clean = c_df.drop(columns=[c for c in cols_to_drop if c in c_df.columns])
        
        v_array = v_clean.astype("float32").values
        c_array = c_clean.astype("float32").values
        
        self.signal_2ch = np.stack((v_array, c_array), axis=1)
        
        self.desc = d_df[descriptors_name].astype("float32").values
        self.signal_tensor = torch.tensor(self.signal_2ch, dtype=torch.float32)
        self.desc_tensor = torch.tensor(self.desc, dtype=torch.float32)

    def __len__(self):
        return len(self.signal_tensor)

    def __getitem__(self, idx):
        return {
            "signal_1d": self.signal_tensor[idx],
            "features": self.desc_tensor[idx]
        }
    

    @staticmethod
    def restore_2d_curve(generated_tensor):
        """
        Restores 2D from generated tenzor
        """
        if isinstance(generated_tensor, torch.Tensor):
            generated_tensor = generated_tensor.detach().cpu().numpy()
        
        voltage = generated_tensor[0, :]
        current = generated_tensor[1, :]
        
        return voltage, current