# data_preprocessor.py
import numpy as np
import pandas as pd
import pywt


class DataPreprocessor:
    """Preprocess raw data: interpolate, validate, clean"""
    
    def __init__(self, raw_extractor, num_cycle, inhibitor_name, use_wavelet=False):
        """
        Args:
            raw_extractor: RawDataExtractor object
            num_cycle: list of cycles
            inhibitor_name: inhibitor name or "all"
            use_wavelet: use wavelet
        """
        self.raw_extractor = raw_extractor
        self.num_cycle = num_cycle
        self.inhibitor_name = inhibitor_name
        self.use_wavelet = use_wavelet
        
        self.voltage = None
        self.current = None
        self.metadata = None
        
        self._preprocess()
    
    def _preprocess(self):
        
        self.current = pd.concat([
            pd.DataFrame(self.raw_extractor.current_raw),
            self.raw_extractor.metadata
        ], axis=1)
        
        self.voltage = pd.concat([
            pd.DataFrame(self.raw_extractor.voltage_raw),
            self.raw_extractor.metadata
        ], axis=1)
        
        self._filter_by_cycle()
        self._filter_by_inhibitor()
        
        self.metadata = {
            'Num_of_Cycle': self.current['Num_of_Cycle'].copy(),
            'ppm': self.current['ppm'].copy(),
            'Inhibitor': self.current['Inhibitor'].copy(),
        }
        
        self.current = self.current.drop(columns=['ppm', 'Num_of_Cycle', 'Inhibitor']) * 1000
        self.voltage = self.voltage.drop(columns=['ppm', 'Num_of_Cycle', 'Inhibitor'])
        
        if self.use_wavelet:
            voltage_arrays = [self.voltage.iloc[i].values for i in range(len(self.voltage))]
            current_arrays = [self.current.iloc[i].values for i in range(len(self.current))]
            
            voltage_interpolated = self._wavelet_interpolate(voltage_arrays)
            current_interpolated = self._wavelet_interpolate(current_arrays)
            
            self.voltage = pd.DataFrame(voltage_interpolated)
            self.current = pd.DataFrame(current_interpolated)

    
    def _wavelet_interpolate(self, data, wavelet='db4', level=2):
        interpolated = []
        
        for arr in data:
            coeffs = pywt.wavedec(arr, wavelet, level=level)
            resampled = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=968)
            interpolated.append(resampled)
        
        return np.array(interpolated)
    
    def _filter_by_cycle(self):
        mask = self.current['Num_of_Cycle'].isin(self.num_cycle)
        self.current = self.current[mask].reset_index(drop=True)
        self.voltage = self.voltage[mask].reset_index(drop=True)
    
    def _filter_by_inhibitor(self):
        if self.inhibitor_name != 'all':
            mask = self.current['Inhibitor'] == self.inhibitor_name
            self.current = self.current[mask].reset_index(drop=True)
            self.voltage = self.voltage[mask].reset_index(drop=True)
