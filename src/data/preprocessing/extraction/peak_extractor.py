import numpy as np
import pandas as pd


class PeakExtractor:
    def __init__(self, voltage, current):

        self.voltage = voltage
        self.current = current

    def __locate_peaks(self, current):

        anodic_peak = np.argmax(current)
        cathodic_peak = np.argmin(current)

        return anodic_peak, cathodic_peak

    def __extract_peak_height_and_area(self, current, anodic_peak, cathodic_peak):

        anodic_height = current[anodic_peak]
        cathodic_height = current[cathodic_peak]

        #anodic_area = np.trapz(current[max(0, anodic_peak-10):min(len(current), anodic_peak+10)])
        #cathodic_area = np.trapz(-current[max(0, cathodic_peak-10):min(len(current), cathodic_peak+10)])
        segment_anodic = current[max(0, anodic_peak-10):min(len(current), anodic_peak+10)]
        anodic_area = np.trapezoid(segment_anodic, dx=1)

        segment_cathodic = -current[max(0, cathodic_peak-10):min(len(current), cathodic_peak+10)]
        cathodic_area = np.trapezoid(segment_cathodic, dx=1)
        
        return anodic_height, cathodic_height, anodic_area, cathodic_area

    def __extract_peak_positions(self, voltage, anodic_peak, cathodic_peak):

        anodic_position = voltage[anodic_peak]
        cathodic_position = voltage[cathodic_peak]

        return anodic_position, cathodic_position

    def __extract_half_wave_potential(self, anodic_position, cathodic_position):

        half_wave_potential = anodic_position + cathodic_position / 2

        return half_wave_potential

    def __extract_delta_E(self, anodic_position, cathodic_position):
        delta_E = abs(anodic_position - cathodic_position)
        return delta_E


    def __extract_data_from_VAC(self, voltage, current):
        anodic_peak, cathodic_peak = self.__locate_peaks(current)
        anodic_height, cathodic_height, anodic_area, cathodic_area = self.__extract_peak_height_and_area(current, anodic_peak, cathodic_peak)
        anodic_position, cathodic_position = self.__extract_peak_positions(voltage, anodic_peak, cathodic_peak)
        half_wave_potentials = self.__extract_half_wave_potential(anodic_position, cathodic_position)
        delta_E = self.__extract_delta_E(anodic_position, cathodic_position)
        
        results = {
            'Anodic Peak Position (V)': anodic_position,
            'Anodic Peak Height (A)': anodic_height,
            'Anodic Peak Area': anodic_area,
            'Cathodic Peak Position (V)': cathodic_position,
            'Cathodic Peak Height (A)': cathodic_height,
            'Cathodic Peak Area': cathodic_area,
            'Half-Wave Potential (V)': half_wave_potentials,
            'Delta E (V)': delta_E
        }

        return results
    

    def get_data(self):

        analyzed_data = {
        'Anodic Peak Position (V)': [],
        'Anodic Peak Height (A)': [],
        'Anodic Peak Area': [],
        'Cathodic Peak Position (V)': [],
        'Cathodic Peak Height (A)': [],
        'Cathodic Peak Area': [],
        'Half-Wave Potential (V)': [],
        'Delta E (V)': []
        }

        for i in range(self.voltage.shape[0]):
            results = self.__extract_data_from_VAC(self.voltage.iloc[i], self.current.iloc[i])
            for key in results.keys():
                analyzed_data[key].append(results[key])

        analyzed_data = pd.DataFrame(analyzed_data)

        self.volt_final = pd.DataFrame()

        vol = pd.DataFrame([volt[1] for volt in self.voltage.items()])

        cur = pd.DataFrame([curr[1] for curr in self.current.items()])

        return analyzed_data, vol, cur