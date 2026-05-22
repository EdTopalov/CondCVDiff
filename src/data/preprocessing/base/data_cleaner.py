# data_cleaner.py
import numpy as np
import pandas as pd


class DataCleaner:
    """Remove noisy tails and interpolate"""
    
    def __init__(self, voltage, current):
        """
        Args:
            voltage: DataFrame 
            current: DataFrame 
        """
        self.voltage = voltage.copy()
        self.current = current.copy()
        
        self._clean()
    
    def _clean(self):
        self._remove_tails()
    
    def _remove_tails(self):
        def _fill_voltage(voltage_df):
            voltage_new = voltage_df.copy()
            volt_increase = 0.008333999
            
            for index, element in voltage_new.iterrows():
                for i in range(element.shape[0]):
                    if pd.isna(element[i]):
                        voltage_new.iloc[index, i] = (
                            voltage_new.iloc[index, i-1] + 
                            volt_increase * np.sign(voltage_new.iloc[index, i-1])
                        )
            
            return voltage_new
        
        indices_to_nan = [(150, 333), (634, 815), (939, 967)]
        
        voltage_new, current_new = [], []
        
        for index, volt_new in _fill_voltage(self.voltage).iterrows():
            curr_new = self.current.iloc[index].copy()
            
            for start, end in indices_to_nan:
                curr_new.iloc[start:end] = np.nan
            
            curr_new = curr_new.interpolate()
            
            voltage_new.append(volt_new)
            current_new.append(curr_new)
        
        self.voltage = pd.DataFrame(voltage_new).reset_index(drop=True)
        self.current = pd.DataFrame(current_new).reset_index(drop=True)
        
        self.voltage.columns = range(self.voltage.shape[1])
        self.current.columns = range(self.current.shape[1])
