import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fft import rfftfreq
from scipy.fftpack import rfft
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import signal, stats
from typing import Dict, List, Tuple

class Preprocessor:
    def __init__(self, health_threshold=0.2, input_dim=128, proportion=0.10, window_size=128, stride=64, scalar='standard'):
        self.health_threshold = health_threshold
        self.input_dim = input_dim
        self.proportion = proportion
        self.window_size = window_size
        self.stride = stride
        if scalar == 'standard':
            self.scalar = StandardScaler()
        elif scalar == 'min_max':
            self.scalar = MinMaxScaler()

         
    @staticmethod
    def create_file_list(test: Path):
        return sorted([f for f in test.iterdir() if f.is_file()])

    @staticmethod
    def get_test_run_time(files, test_num = 1):
        if test_num == 1: # Test one has different timing
            return len(files[:44]) * 5 + len(files[44:]) * 10
        else:
            return len(files) * 10

    @staticmethod 
    def get_time_until_failure(total_test_time, file_idx, test_num):
        if test_num == 1: # test one has different timing
            if file_idx < 44:
                return total_test_time - file_idx * 5
            else:
                return total_test_time - (44 * 5 + (file_idx - 44) * 10)
        else:
            return total_test_time - file_idx * 10

    @staticmethod  
    def get_degradation(time_until_failure, total_test_time):
        return 1 - (time_until_failure / total_test_time) ** 2

    def get_indices(self, file_list):
        n_files = len(file_list)
        indices = np.linspace(0, n_files-1, int(n_files*self.proportion))
        return indices.astype(int)
    
    def extract_time_features(self, window: np.ndarray) -> Dict[str, float]:
        features = {}

        features['mean'] = np.mean(window)
        features['std'] = np.std(window)
        features['var'] = np.var(window)
        features['rms'] = np.sqrt(np.mean(window ** 2))
        features['peak'] = np.max(np.abs(window))
        features['ptp'] = np.ptp(window)

        mean_abs = np.mean(np.abs(window))

        if mean_abs != 0:
            features['shape_factor'] = features['rms'] / mean_abs
            features['impulse_factor'] = features['peak'] / mean_abs
        else:
            features['shape_factor'] = 0  
            features['impulse_factor'] = 0

        if features['rms'] != 0:
            features['crest_factor'] = features['peak'] / features['rms']
        else:
            features['crest_factor'] = 0

        features['kurtoses'] = stats.kurtosis(window)
        features['skewness'] = stats.skew(window)

        sqrt_mean = np.mean(np.sqrt(np.abs(window)))
        if sqrt_mean != 0:
            features['clearance_factor'] = features['peak'] / (sqrt_mean ** 2)
        else:
            features['clearance_factor'] = 0

        features['energy'] = np.sum(window**2)

        zero_crossings = np.where(np.diff(np.sign(window)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(window)

        features['percentile_25'] = np.percentile(window, 25)
        features['percentile_50'] = np.percentile(window, 50)
        features['percentile_75'] = np.percentile(window, 75)

        return features
    
    def extract_frequency_features(self, window: np.ndarray) -> Dict[str, float]:
        features = {}
        
        # FFT
        fft_vals = rfft(window)
        fft_mag = np.abs(fft_vals)
        freqs = rfftfreq(len(window), 1/self.config.sampling_rate)
        
        # Basic spectral statistics
        features['fft_mean'] = np.mean(fft_mag)
        features['fft_std'] = np.std(fft_mag)
        features['fft_max'] = np.max(fft_mag)
        features['fft_sum'] = np.sum(fft_mag)
        
        # Dominant frequencies (top 5)
        top_indices = np.argsort(fft_mag)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            features[f'dominant_freq_{i+1}'] = freqs[idx]
            features[f'dominant_freq_{i+1}_magnitude'] = fft_mag[idx]
        
        # Spectral centroid and spread
        if np.sum(fft_mag) != 0:
            spectral_centroid = np.sum(freqs * fft_mag) / np.sum(fft_mag)
            features['spectral_centroid'] = spectral_centroid
            features['spectral_spread'] = np.sqrt(
                np.sum((freqs - spectral_centroid)**2 * fft_mag) / np.sum(fft_mag)
            )
        else:
            features['spectral_centroid'] = 0
            features['spectral_spread'] = 0
        
        # Spectral entropy
        if np.sum(fft_mag) > 0:
            psd_norm = fft_mag / np.sum(fft_mag)
            psd_norm = psd_norm[psd_norm > 0]  # Remove zeros for log
            features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
        else:
            features['spectral_entropy'] = 0
        
        # Band power features
        band_edges = [0, 200, 500, 1000, 2000, 4000, 8000, 10000]  # Hz
        for i in range(len(band_edges) - 1):
            band_mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            band_power = np.sum(fft_mag[band_mask]**2)
            features[f'band_power_{band_edges[i]}_{band_edges[i+1]}Hz'] = band_power
            
            # Relative band power
            if np.sum(fft_mag**2) > 0:
                features[f'band_power_ratio_{band_edges[i]}_{band_edges[i+1]}Hz'] = (
                    band_power / np.sum(fft_mag**2)
                )
            else:
                features[f'band_power_ratio_{band_edges[i]}_{band_edges[i+1]}Hz'] = 0

    def get_windows(self, sensor_data) -> list[np.array]: 
        windows = []

        for i in range(0, len(sensor_data) - self.window_size, self.stride):
            window = sensor_data[i:i+self.window_size]
            features = self.extract_time_features(window)
            features_frame = pd.DataFrame(features)
            windows.append(features_frame)

        return windows

    def get_scaled_data(self, data) -> np.array:
        scaled_data = self.scalar.fit_transform(data)
        return scaled_data
    
    def bearing_test_data_pipeline(self, files, total_test_time, indices, test_num):
        """
        The simple data pipeline aims to treat all of the data from the tests as though there is just one bearing.
        Inputs: 
            files: a list of all files in a test directory
            total_test_time: The total amount of time that the test ran, used to calculate degradation
            indices: The sample of files from that test that is being used
            test_num: used for time until failure and degradation calculation, the first test had several quirks,
            see NASA Bearing vibrational dataset documentation

        Returns:
            all_healthy_windows: np.array containing healthy bearing data only
            all_damaged_windows: np.array containing windows that are considered 'unhealthy'
        """
        all_healthy_windows = []
        all_damaged_windows = []

        for file_idx in indices:
            # Get the time until failure
            time_until_failure = self.get_time_until_failure(total_test_time, file_idx, test_num)

            # Using a custom degradation calculation to capture healthy bearing data 
            degradation = self.get_degradation(time_until_failure, total_test_time)
            
            # Raw data from file
            data = np.loadtxt(files[file_idx])

            for col_idx in range(data.shape[1]):
                # Grab each column from the raw data
                sensor_data = data[:, col_idx]

                # Window the sensor_data
                windows = self.get_windows(sensor_data)

                # Split the data into the healthy and unhealthy sets, the unhealthy set is our test set
                # Unhealthy in this case is considered the "Second half" of the bearings life. See get_degradation for details
                if degradation <= self.health_threshold:
                    all_healthy_windows.extend(windows)
                else:
                    all_damaged_windows.extend(windows)

        return all_healthy_windows, all_damaged_windows