import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import kurtosis, skew

class Preprocessor:


    def extract_time_features(self, window: np.ndarray, features: Dict[str, List[float]]) -> None: 

            features = {}
            
            # Basic statistics
            features['mean'].append(np.mean(window)) 
            features['std'].append(np.std(window))  
            features['var'].append(np.var(window))  
            features['rms'].append(np.sqrt(np.mean(window**2)))  
            features['peak'].append(np.max(np.abs(window)))  
            features['peak_to_peak'].append(np.ptp(window))  
            
            # Shape factors
            mean_abs = np.mean(np.abs(window))
            if mean_abs != 0:
                features['shape_factor'].append(features['rms'] / mean_abs)  
                features['impulse_factor'].append(features['peak'] / mean_abs)  
            else:
                features['shape_factor'].append(0)
                features['impulse_factor'].append(0)
            
            if features['rms'] != 0:
                features['crest_factor'].append(features['peak'] / features['rms'])
            else:
                features['crest_factor'].append(0)
            
            # Higher-order statistics
            features['kurtosis'].append() = kurtosis(window)
            features['skewness'].append() = skew(window)
            
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
