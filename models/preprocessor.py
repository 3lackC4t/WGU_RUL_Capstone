import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SensorPreprocessor:
    def __init__(self, tests=None, proportion=0.1, window_size=512, stride=256, init_build=True):
        self.window_size = window_size
        self.stride = stride 

        if not init_build:
            self.tests = None
        else:
            self.tests = tests
        
        self.test_data = {}
        
        if init_build:
            for idx, test in enumerate(tests):
                files = self.create_file_list(test)
                total_time = self.get_test_run_time(files, idx + 1)
                indices = self.get_indices(files, proportion)
                self.test_data[f'test_{idx + 1}'] = {
                    'sensor_data_files': files,
                    'total_run_time': total_time,
                    'indices': indices,
                    'sensor_data': self.get_bulk_arrays(files, total_time, indices, idx+1)
                }
         
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

    @staticmethod 
    def get_indices(file_list, proportion):
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

        return bulk_data
    
    def simple_data_pipeline(self, files, total_test_time, indices, test_num):
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

            for col in data.columns:
                # Grab each column from the raw data
                sensor_data = data[:, col]

                # Window the sensor_data
                windows = []

                # Simple sliding window
                for i in range(0, len(sensor_data) - self.window_size, self.stride):
                    window = sensor_data[i:i+self.window_size]
                    windows.append(window)

                # Split the data into the healthy and unhealthy sets, the unhealthy set is our test set
                # Unhealthy in this case is considered the "Second half" of the bearings life. See get_degradation for details
                if degradation <= 0.5:
                    all_healthy_windows.append(windows)
                else:
                    all_damaged_windows.append(windows)

        return np.array(all_healthy_windows), np.array(all_damaged_windows)

    
    def apply_scaling(self, data: pd.DataFrame, scalar_type: str = 'standard'):
        sensor_columns = [col for col in data.columns if col.startswith("sensor_")]

        if scalar_type == 'standard':
            scaler = StandardScaler()
        elif scalar_type == 'minmax':
            scaler = MinMaxScaler()

        scaled_data = data.copy()
        scaled_data[sensor_columns] = scaler.fit_transform(data[sensor_columns])

        return scaled_data

    def get_feature_windows(self, arr):
        X_windows = []
        print(f"creating {len(arr) // self.window_size} windows")

        for i in range(0, len(arr) - self.window_size, self.stride):
            X_window = arr[i:i+self.window_size]

            X_windows.append(X_window)
        
        X = np.array(X_windows)
        return X
    
    def get_target_windows(self, arr):
        y_windows = []
        print(f"creating {len(arr) // self.window_size} windows")

        for i in range(0, len(arr) - self.window_size, self.stride):
            y_window = arr[i + self.window_size -1]
            y_windows.append(y_window)
        
        y = np.array(y_windows)
        return y

    def apply_scaling_on_input(self, data: pd.DataFrame, scalar_type: str = 'standard'):
        if scalar_type == 'standard':
            scaler = StandardScaler()
        elif scalar_type == 'minmax':
            scaler = MinMaxScaler()

        scaled_data = data.copy()
        scaled_data = scaler.fit_transform(data)

        return scaled_data

    def get_cleaned_input(self, file_object, file_type, rpm=1000):
        if file_type == 'txt':
            raw_data = pd.DataFrame(np.loadtxt(file_object, comments='#', delimiter=None))
            all_sensors = []
            print(f"Data Type: {type(raw_data)}")
            print(f"Data Shape: {raw_data.shape}")
            if len(raw_data.columns) > 1:
                for col in raw_data.columns:
                    single_raw_data = raw_data.loc[:, col].to_frame()
                    
                    print(f"Data Type: {type(single_raw_data)}")
                    print(f"Data Shape: {single_raw_data.shape}")
                    
                    scaled_single_raw_data = self.apply_scaling_on_input(single_raw_data)
                    windows = []
                    for i in range(0, len(scaled_single_raw_data) - self.window_size, self.stride):
                        window = scaled_single_raw_data[i:i + self.window_size]
                        windows.append(window)
                    
                    all_sensors.append(np.array(windows))

            else:
                scaled_data = self.apply_scaling_on_input(raw_data)
                windows = []
                for i in range(0, len(scaled_data) - self.window_size, self.stride):
                    window = scaled_data[i:i + self.window_size]
                    windows.append(window)

                all_sensors.append(np.array(windows))

            return all_sensors

