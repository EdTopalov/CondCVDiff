import pandas as pd
from src.data.preprocessing.base.raw_data_extractor import RawDataExtractor
from src.data.preprocessing.base.data_preprocessor import DataPreprocessor
from src.data.preprocessing.base.data_cleaner import DataCleaner
from src.data.preprocessing.augmentation.descriptor_merger import DescriptorMerger
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

class Pipeline:    
    def __init__(self, num_cycle, test_inhibitor=None, norm_feat=False, use_wavelet=False, flip_the_peak: bool = True, use_cycle_feat=True):
        """
        Args:
            num_cycle: list of num cycles
            test_inhibitor: Inhibitor name for test
            norm_feat: normalize mol features
            use_wavelet: add wavelet to CV
        """
        self.test_inhibitor = test_inhibitor
        
        # ============ STEP 1: Download raw data ============
        raw_extractor = RawDataExtractor(split="all")
        
        # ============ STEP 2: Preprocess ============
        preprocessor = DataPreprocessor(
            raw_extractor,
            num_cycle=num_cycle,
            inhibitor_name="all",
            use_wavelet=use_wavelet
        )
        
        # ============ STEP 3: Clean from artefacts ============
        cleaner = DataCleaner(preprocessor.voltage, preprocessor.current)
        self.all_voltage = cleaner.voltage.copy()
        self.all_current = cleaner.current.copy()
        self.all_metadata = pd.DataFrame(preprocessor.metadata)
        
        # ============ STEP 4: Split to Train / Test ============
        metadata_df = pd.DataFrame(preprocessor.metadata)
                
        if self.test_inhibitor:
            train_mask = metadata_df['Inhibitor'] != self.test_inhibitor
            test_mask = metadata_df['Inhibitor'] == self.test_inhibitor
        else:
            train_mask = pd.Series(True, index=metadata_df.index)
            test_mask = pd.Series(False, index=metadata_df.index)
            
        self.train_voltage = cleaner.voltage[train_mask].reset_index(drop=True)
        self.train_current = cleaner.current[train_mask].reset_index(drop=True)
        train_metadata = metadata_df[train_mask].reset_index(drop=True)
        
        self.cycle_scaler = MinMaxScaler()
        train_metadata['Num_of_Cycle_Norm'] = self.cycle_scaler.fit_transform(train_metadata[['Num_of_Cycle']])
        self.train_metadata = train_metadata

        self.train_metadata = train_metadata #new

        if self.test_inhibitor:
            self.test_voltage = cleaner.voltage[test_mask].reset_index(drop=True)
            self.test_current = cleaner.current[test_mask].reset_index(drop=True)
            test_metadata = metadata_df[test_mask].reset_index(drop=True)

            test_metadata['Num_of_Cycle_Norm'] = self.cycle_scaler.transform(test_metadata[['Num_of_Cycle']])
            self.test_metadata = test_metadata#new
        else:
            self.test_voltage, self.test_current, test_metadata = None, None, None
        
        
        # ========================= Scale signals =============================
        if flip_the_peak:
            self.mid_idx = self.train_current.shape[1] // 2
            
            self.train_current.iloc[:, self.mid_idx:] = self.train_current.iloc[:, self.mid_idx:] * -1.0
            self.train_voltage = self.train_voltage

            if self.test_inhibitor is not None:
                self.test_voltage = self.test_voltage
                self.test_current.iloc[:, self.mid_idx:] = self.test_current.iloc[:, self.mid_idx:] * -1.0
            else:
                self.test_voltage, self.test_current = None, None
        else:
            ...

            
        # ============ STEP 5: Add descriptors ============
        merger = DescriptorMerger(normalize=norm_feat)
        
        self.train_analyzed_data = merger.fit_transform(train_metadata)
        
        if self.test_inhibitor:
            self.test_analyzed_data = merger.transform(test_metadata)
        else:
            self.test_analyzed_data = None