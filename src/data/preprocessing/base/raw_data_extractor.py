# raw_data_extractor.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.paths import get_project_path


class RawDataExtractor:
    """Reads .edf files and return raw data"""
    
    def __init__(self, split="all"):
        """
        Args:
            split: "all", "train", или "test"
        """
        self.split = split
        self.voltage_raw = None
        self.current_raw = None
        self.metadata = None
        
        self._extract()
    
    def _extract(self):
        """Main method - initialize all raw data"""
        list_files = self._collect_files()
        
        current, voltage, conc, inh, num_of_cycle_df = [], [], [], [], []
        
        for file in list_files:
            with open(file, encoding='latin-1') as f:
                data = f.readlines()
            
            current_temp, volt_temp = [], []
            num_of_cycle = 0
            
            for line in data:
                values = line.strip().split()
                
                if len(values) == 4:
                    current_temp.append(float(values[3]))
                    volt_temp.append(float(values[2]))
                
                elif values[0] == 'de':
                    num_of_cycle_df.append(num_of_cycle)
                    num_of_cycle += 1
                    
                    current.append(current_temp)
                    voltage.append(volt_temp)
                    file_parts = file.strip().split(os.sep)
                    conc.append(int(file_parts[-2].split()[0]))
                    inh.append(file_parts[-3].split()[0])
                    
                    current_temp, volt_temp = [], []
        
        drop_idx = [i for i in range(len(current)) if len(current[i]) < 900]
        
        current = [cur for i, cur in enumerate(current) if i not in drop_idx]
        voltage = [val for i, val in enumerate(voltage) if i not in drop_idx]
        conc = [ppm for i, ppm in enumerate(conc) if i not in drop_idx]
        inh = [inhib for i, inhib in enumerate(inh) if i not in drop_idx]
        num_of_cycle_df = [cyc for i, cyc in enumerate(num_of_cycle_df) if i not in drop_idx]
        
        self.voltage_raw = voltage
        self.current_raw = current
        self.metadata = pd.DataFrame({
            "ppm": conc,
            "Inhibitor": inh,
            "Num_of_Cycle": num_of_cycle_df
        })
    
    def _collect_files(self):
        """Collect list of files with split"""
        list_files = []
        for root, dirs, files in os.walk(os.path.join(get_project_path(), 'data', 'experimental_data')):
            for file in files:
                if file.endswith('.edf'):
                    list_files.append(os.path.join(root, file))
        
        if self.split == "all":
            return list_files
        elif self.split == "train":
            list_files, _ = train_test_split(list_files, test_size=0.1, random_state=2683)
            return list_files
        elif self.split == "test":
            _, list_files = train_test_split(list_files, test_size=0.1, random_state=2683)
            return list_files
