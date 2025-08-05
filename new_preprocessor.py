import numpy as np
import pandas as pd
from pathlib import Path

from settings import Settings

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SensorPreprocessor:
    def __init__(self, tests, proportion=0.10, window_size=32, stride=16, final_output_dim=(32, 1), init_build=True):
        self.final_output_dim = final_output_dim
        self.window_size = window_size
        self.stride = stride 
        self.tests = tests
        
        if init_build:
            test_one_files = self.create_file_list(self.tests[0])
            test_two_files = self.create_file_list(self.tests[1])
            test_three_files = self.create_file_list(self.tests[2])

            test_one_time = self.get_test_run_time(test_one_files, 1)
            test_two_time = self.get_test_run_time(test_two_files, 2)
            test_three_time = self.get_test_run_time(test_three_files, 3)

            test_one_indices = self.get_indices(test_one_files, proportion)
            test_two_indices = self.get_indices(test_two_files, proportion)
            test_three_indices = self.get_indices(test_three_files, proportion) 

            self.tests = {
                "test_1": {
                    'sensor_data_files': test_one_files,
                    'total_run_time': test_one_time,
                    'indices': test_one_indices,
                    'sensor_data': self.get_bulk_arrays(
                        test_one_files, 
                        test_one_time, 
                        test_one_indices, 
                        1
                    ),
                },
                "test_2": {
                    'sensor_data_files': test_two_files,
                    'total_run_time': test_two_time,
                    'indices': test_two_indices,
                    'sensor_data': self.get_bulk_arrays(
                        test_two_files, 
                        test_two_time, 
                        test_two_indices, 
                        2
                    ),
                },
                "test_3": {
                    'sensor_data_files': test_three_files,
                    'total_run_time': test_three_time,
                    'indices': test_three_indices,
                    'sensor_data': self.get_bulk_arrays(
                        test_three_files, 
                        test_three_time, 
                        test_three_indices, 
                        3
                    ),
                }
            }
            self.tests['test_1']['X_windows'], self.tests['test_1']['y_windows'] = self.get_windows(self.tests['test_1']['sensor_data'])
            self.tests['test_2']['X_windows'], self.tests['test_2']['y_windows'] = self.get_windows(self.tests['test_2']['sensor_data'])
            self.tests['test_3']['X_windows'], self.tests['test_3']['y_windows'] = self.get_windows(self.tests['test_3']['sensor_data'])

    @staticmethod
    def create_file_list(test: Path):
        return sorted([f for f in test.iterdir()])

    @staticmethod
    def get_test_run_time(files, test_num = 1):
        if test_num == 1:
            return len(files[:44]) * 5 + len(files[44:]) * 10
        else:
            return len(files) * 10

    @staticmethod 
    def get_time_until_failure(total_test_time, file_idx, test_num):
        if test_num == 1:
            if file_idx < 44:
                return total_test_time - file_idx * 5
            else:
                return total_test_time - (44 * 5 + (file_idx - 44) * 10)
        else:
            return total_test_time - file_idx * 10

    @staticmethod  
    def get_degradation(time_until_failure, total_test_time):
        return 1 - (time_until_failure / total_test_time) ** 2
    
    def get_indices(self, file_list, proportion):
        n_files = len(file_list)
        indices = np.linspace(0, n_files-1, int(n_files*proportion))
        return indices.astype(int)

    def get_bulk_arrays(self, file_list, total_test_time, indices, test_num):
        print(f"Creating bulk sensor data for test {test_num}")
        data_frames = []
        for index in indices:
            print(f"Getting sample {index} of {len(indices)}")
            time_until_failure = self.get_time_until_failure(total_test_time, index, test_num)
            degradation = self.get_degradation(time_until_failure, total_test_time)

            raw_data = pd.DataFrame(np.loadtxt(file_list[index]))
            n_sensors = raw_data.shape[1]
            raw_data.columns = [f'sensor_{i+1}' for i in range(n_sensors)]
            
            print(f"Creating sample array, appending degradation")
            deg_arr = [degradation for _ in range(len(raw_data))] 
            raw_data['degradation'] = deg_arr
            data_frames.append(raw_data)
            
        bulk_data = pd.concat(data_frames, ignore_index=True)

        print(f"complete with test {test_num}")

        scaled_data, scaler = self.apply_scaling(bulk_data, 'standard')

        return scaled_data
    
    def apply_scaling(self, data: pd.DataFrame, scalar_type: str = 'standard'):
        sensor_columns = [col for col in data.columns if col.startswith("sensor_")]

        if scalar_type == 'standard':
            scaler = StandardScaler()
        elif scalar_type == 'minmax':
            scaler = MinMaxScaler()

        scaled_data = data.copy()
        scaled_data[sensor_columns] = scaler.fit_transform(data[sensor_columns])

        return scaled_data, scaler

    def get_windows(self, arr):
        X_windows, y_windows = [], []
        print(f"creating {len(arr) // self.window_size} windows")

        for i in range(0, len(arr) - self.window_size, self.stride):
            X_window = arr.iloc[i:i+self.window_size]
            y_window = arr.iloc[i + self.window_size -1]

            X_windows.append(X_window)
            y_windows.append(y_window)
        
        X = np.array(X_windows).reshape(-1, self.window_size, 1)
        y = np.array(y_windows)

        return X, y
    
    def get_sensor_frame(self, test_num):
        sensor_training_data = []

