import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq

from models.bearing_config import BearingConfig

class NASABearingPreprocessor:
    def __init__(
        self, window_size: int = 2048, 
        overlap: float = 0.5,
        scaler_type: str = 'standard',
        health_threshold: float = 0.6,
        proportion: float = 0.1,
        extract_features: bool = True
    ):
        self.window_size = window_size
        self.overlap = overlap
        self.stride = int(window_size * (1 - overlap))
        self.proportion = proportion
        self.scalar_type = scaler_type
        self.health_threshold = health_threshold
        self.reference_threshold = None
        self.extract_features = extract_features

        self.scalar = self._get_scalar(scaler_type)
        self.is_fitted = False

        self.config = BearingConfig()
        
        self.feature_names = []

    def _get_scalar(self, scaler_type: str) -> object:
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        return scalers.get(scaler_type, StandardScaler())

    def parse_bearing_data(self, file_path: Path, test_num: int) -> Dict[str, np.ndarray]:
        data = np.loadtxt(file_path)

        metadata = self.get_test_metadata(test_num)
        bearing_data = {}

        if test_num == 1:

            for bearing_idx in range(metadata['num_bearings']):
                bearing_name = f"bearing_{bearing_idx + 1}"
                bearing_data[bearing_name] = {}
                for sensor_idx in range(metadata['sensors_per_bearing']):
                    col_idx = bearing_idx * metadata['sensors_per_bearing'] + sensor_idx
                    sensor_name = f"sensor_{sensor_idx + 1}"
                    bearing_data[bearing_name][sensor_name] = data[:, col_idx]
        
        else:
            for bearing_idx in range(metadata['num_bearings']):
                bearing_name = f"bearing_{bearing_idx + 1}"
                bearing_data[bearing_name] = {
                    "sensor_1": data[:, bearing_idx]
                }

        return bearing_data
    
    def create_file_list(self, test_path: Path) -> List[Path]:
        return sorted([f for f in test_path.iterdir() if f.is_file()])
    
    def get_test_metadata(self, test_num) -> Dict[str, int]:
        if test_num == 1:
            return {
                'num_bearings': self.config.test_1_bearings,
                'sensors_per_bearing': self.config.test_1_sensors_per_bearing,
                'total_channels': self.config.test_1_bearings * self.config.test_1_sensors_per_bearing
            }
        else:
            return {
                'num_bearings': self.config.test_2_bearings,
                'sensors_per_bearing': self.config.test_2_sensors_per_bearing,
                'total_channels': self.config.test_2_bearings * self.config.test_2_sensors_per_bearing
            }
        
    def get_time_until_failure(self, file_idx: int, total_files: int, test_num: int) -> float:

        if test_num == 1:
            if file_idx < self.config.test_1_early_file_cutoff:
                time_elapsed = file_idx * self.config.test_1_early_interval
            else:
                time_elapsed = (self.config.test_1_early_file_cutoff * self.config.test_1_early_interval
                                + (file_idx - self.config.test_1_early_file_cutoff) * self.config.test_1_late_interval)
                
            if total_files <= self.config.test_1_early_file_cutoff:
                total_time = total_files * self.config.test_1_early_interval
            else:
                total_time = (self.config.test_1_early_file_cutoff * self.config.test_1_early_interval + 
                              (total_files - self.config.test_1_early_file_cutoff) * self.config.test_1_late_interval)
                
        else:
            time_elapsed = file_idx * self.config.standard_interval
            total_time = total_files * self.config.standard_interval

        return total_time - time_elapsed
    
    def calculate_degradation(self, time_until_failure: float, total_test_time: float, degradation_calculation_method: str) -> float:

        if degradation_calculation_method == 'linear':
            degradation = time_until_failure / total_test_time
        elif degradation_calculation_method == 'semi_linear':
            if time_until_failure <= 0.5 * total_test_time:
                degradation = 1
            else:
                degradation = time_until_failure / total_test_time
        elif degradation_calculation_method == 'non_linear':
            degradation = 1 - (time_until_failure / total_test_time) ** 2
        
        return degradation
    
    def create_windows(self, signal_data: np.ndarray) -> np.ndarray:

        windows = []
        for i in range(0, len(signal_data) - self.window_size + 1, self.stride):
            windows.append(signal_data[i:i+self.window_size])
        
        return windows
    
    def extract_time_features(self, window: np.ndarray) -> Dict[str, float]:

        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(window)
        features['std'] = np.std(window)
        features['var'] = np.var(window)
        features['rms'] = np.sqrt(np.mean(window**2))
        features['peak'] = np.max(np.abs(window))
        features['peak_to_peak'] = np.ptp(window)
        
        # Shape factors
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
        
        # Higher-order statistics
        features['kurtosis'] = kurtosis(window)
        features['skewness'] = skew(window)
        
        # Clearance factor
        sqrt_mean = np.mean(np.sqrt(np.abs(window)))
        if sqrt_mean != 0:
            features['clearance_factor'] = features['peak'] / (sqrt_mean ** 2)
        else:
            features['clearance_factor'] = 0
        
        # Energy
        features['energy'] = np.sum(window**2)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(window)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(window)
        
        # Percentiles
        features['percentile_25'] = np.percentile(window, 25)
        features['percentile_50'] = np.percentile(window, 50)
        features['percentile_75'] = np.percentile(window, 75)
        
        return features
    
    def extract_frequency_features(self, window: np.ndarray) -> Dict[str, float]:

        features = {}
        
        # FFT
        fft_vals = rfft(window)
        fft_mag = np.abs(fft_vals)
        freqs = rfftfreq(len(window), 1/self.config.sample_rate)
        
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
        
        # Spectral rolloff
        cumsum = np.cumsum(fft_mag)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = freqs[rolloff_idx[0]]
            else:
                features['spectral_rolloff'] = freqs[-1]
        else:
            features['spectral_rolloff'] = 0
        
        return features
    
    def extract_all_features(self, window: np.ndarray) -> np.ndarray:

        all_features = {}
        all_features.update(self.extract_time_features(window))
        all_features.update(self.extract_frequency_features(window))

        if not self.feature_names:
            self.feature_names = list(all_features.keys())

        return np.array(list(all_features.values()))
    
    def process_signal(self, signal_data: np.ndarray) -> np.ndarray:

        windows = self.create_windows(signal_data)

        if not self.extract_features:
            return windows / (np.max(np.abs(windows)) + 1e-8)
        
        features = []

        for window in windows:
            features.append(self.extract_all_features(window))

        return np.array(features)
    
    def process_bearing_file(
            self,
            file_path: Path,
            test_num: int,
            file_idx: int,
            total_files: int
    ) -> Tuple[str, np.ndarray, float]:
        
        bearing_data = self.parse_bearing_data(file_path, test_num)

        time_until_failure = self.get_time_until_failure(file_idx, total_files, test_num)
        total_test_time = self.get_time_until_failure(0, total_files, test_num)
        degradation = self.calculate_degradation(time_until_failure, total_test_time, "non_linear")
        
        processed_data = {}
        for bearing_name, sensors in bearing_data.items():
            processed_data[bearing_name] = {}
            for sensor_name, signal in sensors.items():
                processed_signal = self.process_signal(signal) 
                processed_data[bearing_name][sensor_name] = processed_signal

        return processed_data, degradation
    
    def process_input_file(self, file_path: Path) -> np.ndarray:
        print(f"Received file at {file_path}")
        data = np.loadtxt(file_path)
        columns = data.shape[1]

        processed_data = {}

        for bearing_idx in range(columns):
            bearing_name = f"bearing_{bearing_idx + 1}"
            signal = data[:, bearing_idx]

            processed_signal = self.process_signal(signal)

            if self.is_fitted:
                processed_signal = self.transform(processed_signal)
            else:
                raise ValueError("Preprocessor must be fitted before processing")
            
            processed_data[bearing_name] = processed_signal

        return processed_data
    
    def process_test_data(
            self,
            test_path: Path,
            test_num: int,
            specific_bearings: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        
        files = self.create_file_list(test_path)
        total_files = len(files)

        if self.proportion < 1.0:
            n_samples = max(1, int(total_files * self.proportion))
            sample_indices = np.linspace(0, total_files - 1, n_samples).astype(int)
        else:
            n_samples = total_files
            sample_indices = np.arange(total_files)

        print(f"Processing test {test_num}: {n_samples}/{total_files} files")

        healthy_data = []
        degraded_data = []

        metadata = {
            'test_num': test_num,
            'total_files': total_files,
            'sampled_files': n_samples,
            'bearings_processed': specific_bearings or 'all',
            'healthy_samples': 0,
            'degraded_samples': 0
        }

        for idx, file_idx in enumerate(sample_indices):
            if idx % 10 == 0:
                print(f"    Processing file {file_idx}/{n_samples}")

            file_path = files[file_idx]
            processed_data, degradation = self.process_bearing_file(
                file_path, test_num, file_idx, total_files
            )

            for bearing_name, sensors in processed_data.items():
                if specific_bearings and bearing_name not in specific_bearings:
                    continue

                for sensor_name, features in sensors.items():
                    if degradation >= self.health_threshold:
                        healthy_data.extend(features)
                        metadata['healthy_samples'] += len(features)
                    else:
                        degraded_data.extend(features)
                        metadata['degraded_samples'] += len(features)

        healthy_array = np.array(healthy_data)
        degraded_array = np.array(degraded_data)

        print(f"    Healthy samples: {len(healthy_array)}, Degraded samples: {len(degraded_array)}")

        return healthy_array, degraded_array, metadata
            
    def process_multiple_tests(
        self,
        test_paths: List[Tuple[Path, int]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:

        all_healthy = []
        all_degraded = []
        all_metadata = []
        
        for test_path, test_num in test_paths:
            print(f"\nProcessing Test {test_num}")
            healthy, degraded, metadata = self.process_test_data(
                test_path, test_num
            )
            
            all_healthy.append(healthy)
            all_degraded.append(degraded)
            all_metadata.append(metadata)
        
        # Combine all data
        combined_healthy = np.vstack(all_healthy) if all_healthy else np.array([])
        combined_degraded = np.vstack(all_degraded) if all_degraded else np.array([])
        
        print(f"\nTotal healthy samples: {len(combined_healthy)}")
        print(f"Total degraded samples: {len(combined_degraded)}")
        
        return combined_healthy, combined_degraded, all_metadata
            
    def fit(self, data: np.ndarray):
        self.scalar.fit(data)
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        return self.scalar.transform(data)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)
    
    def generate_feature_names(self) -> None:
        dummy_window = np.random.randn(self.window_size)
        _ = self.extract_all_features(dummy_window)
    
    def get_feature_dim(self) -> int:
        if self.extract_features:

            if not self.feature_names:
                self.generate_feature_names() 
                
            return len(self.feature_names)
        else:
            return self.window_size
        
    def get_feature_names(self) -> List[str]:
        if not self.extract_features:
            return [f"sample_{i}" for i in range(self.window_size)]
        
        if not self.feature_names:
            self.generate_feature_names()

        return self.feature_names
    
    def save_data(self, file_path: Path, data: np.ndarray) -> None:
        np.save(file_path, data)

    def save_preprocessor(self, file_path: Path) -> None:
        import pickle

        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        state = {
            'scaler': self.scalar,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'stride': self.stride,
            'health_threshold': self.health_threshold,
            'extract_features': self.extract_features,
            'reference_threshold': self.reference_threshold,
            'config': self.config
        }

        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"Preprocessor saved to {file_path}")

    def load_preprocessor(self, file_path: Path):
        import pickle

        with open(file_path, 'rb') as f:
            state = pickle.load(f)

        self.scalar = state['scaler']
        self.feature_names = state['feature_names']
        self.is_fitted = state['is_fitted']
        self.window_size = state['window_size']
        self.overlap = state['overlap']
        self.stride = state['stride']
        self.health_threshold = state['health_threshold']
        self.extract_features = state['extract_features']
        self.reference_threshold = state['reference_threshold']
        self.config = state['config']

        print(f"Preprocessor loaded from {file_path}")

    def load_data(self, file_path) -> np.ndarray:
        reference_mse = np.load(file_path)
        self.calculate_threshold(reference_mse)
    
    def get_error_mse(self, input_a: np.ndarray, input_b: np.ndarray) -> np.ndarray:
        return np.mean((input_a - input_b) ** 2, axis=1)

    def calculate_threshold(self, reference_mse) -> np.ndarray:
        self.reference_threshold = np.percentile(reference_mse, 95) 