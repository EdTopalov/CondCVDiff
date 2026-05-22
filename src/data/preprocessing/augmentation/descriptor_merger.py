import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.utils.paths import get_project_path


class DescriptorMerger:
    """Merge CV metadata with molecular descriptors strictly avoiding Data Leakage"""
    
    def __init__(self, normalize=False):
        """
        Args:
            normalize: if True, will use MinMaxScaler on descriptors and ppm
        """
        self.normalize = normalize
        self.desc_scaler = MinMaxScaler() if normalize else None
        self.ppm_scaler = MinMaxScaler() if normalize else None
        self.is_fitted = False

    def _load_descriptors(self):
        """Load and merge static descriptor files"""
        rdkit = pd.read_csv(
            os.path.join(get_project_path(), "data", "mol_rdkit.csv")
        ).drop(columns=["SMILES"])
        
        dft = pd.read_csv(
            os.path.join(get_project_path(), "data", "mol_dft.csv")
        )
        return pd.merge(rdkit, dft, how="left", on=["Inhibitor"])

    def fit_transform(self, metadata):
        """
        Fits scalers on the provided metadata (Train set) 
        and returns the merged, scaled features.
        """
        res_df = pd.DataFrame(metadata).copy()
        res_df['raw_ppm'] = res_df['ppm'].copy()
        raw_desc = self._load_descriptors()

        train_inhibitors = res_df['Inhibitor'].unique()
        desc_subset = raw_desc[raw_desc['Inhibitor'].isin(train_inhibitors)].copy()

        desc_subset['raw_MolWt'] = desc_subset['MolWt'].copy()

        if self.normalize:
            desc_features = desc_subset.drop(columns=["Inhibitor", "raw_MolWt"])
            scaled_array = self.desc_scaler.fit_transform(desc_features)
            
            scaled_df = pd.DataFrame(
                scaled_array, 
                columns=desc_features.columns, 
                index=desc_subset.index
            )
            desc_subset = pd.concat([desc_subset[['Inhibitor', 'raw_MolWt']], scaled_df], axis=1)

            res_df['ppm'] = self.ppm_scaler.fit_transform(res_df[['ppm']]).ravel()
            self.is_fitted = True

        return res_df.merge(desc_subset, how="left", on="Inhibitor")

    def transform(self, metadata):
        """
        Transforms Test metadata using fitted scalers 
        and returns the merged features.
        """
        if self.normalize and not self.is_fitted:
            raise ValueError("Scalers are not fitted! Call fit_transform on Train data first.")

        res_df = pd.DataFrame(metadata).copy()

        res_df['raw_ppm'] = res_df['ppm'].copy()

        raw_desc = self._load_descriptors()

        test_inhibitors = res_df['Inhibitor'].unique()
        desc_subset = raw_desc[raw_desc['Inhibitor'].isin(test_inhibitors)].copy()
        desc_subset['raw_MolWt'] = desc_subset['MolWt'].copy()
        

        if self.normalize:
            desc_features = desc_subset.drop(columns=["Inhibitor", "raw_MolWt"])
            
            scaled_array = self.desc_scaler.transform(desc_features)
            
            scaled_df = pd.DataFrame(
                scaled_array, 
                columns=desc_features.columns, 
                index=desc_subset.index
            )
            
            desc_subset = pd.concat([desc_subset[['Inhibitor', 'raw_MolWt']], scaled_df], axis=1)
            res_df['ppm'] = self.ppm_scaler.transform(res_df[['ppm']]).ravel()

        return res_df.merge(desc_subset, how="left", on="Inhibitor")