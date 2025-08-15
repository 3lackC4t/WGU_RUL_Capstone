import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Preprocessor:
    def __init__(self, proportion=0.1, window_size=128, stride=64, scalar='standard'):
        self.proportion = proportion
        self.window_size = window_size
        self.stride = stride
        if scalar == 'standard':
            self.scalar = StandardScaler()
        elif scalar == 'min_max':
            self.scalar = MinMaxScaler()

         
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

    def get_indices(self, file_list):
        n_files = len(file_list)
        indices = np.linspace(0, n_files-1, int(n_files*self.proportion))
        return indices.astype(int)
    
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

        return all_healthy_windows, all_damaged_windows