import pandas as pd
from src.data.preprocessing.base.raw_data_extractor import RawDataExtractor
from src.data.preprocessing.base.data_preprocessor import DataPreprocessor
from src.data.preprocessing.base.data_cleaner import DataCleaner
from src.data.preprocessing.augmentation.descriptor_merger import DescriptorMerger

class Pipeline:    
    def __init__(self, num_cycle, test_inhibitor=None, norm_feat=False, use_wavelet=False):
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
        
        if self.test_inhibitor:
            self.test_voltage = cleaner.voltage[test_mask].reset_index(drop=True)
            self.test_current = cleaner.current[test_mask].reset_index(drop=True)
            test_metadata = metadata_df[test_mask].reset_index(drop=True)
        else:
            self.test_voltage, self.test_current, test_metadata = None, None, None

        # ============ STEP 5: Add descriptors ============
        merger = DescriptorMerger(normalize=norm_feat)
        
        self.train_analyzed_data = merger.fit_transform(train_metadata)
        
        if self.test_inhibitor:
            self.test_analyzed_data = merger.transform(test_metadata)
        else:
            self.test_analyzed_data = None