import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Preprocessor:
    def __init__(self, tests: list[Path]):
        self.tests = [
            self.create_file_list(tests[0]),
            self.create_file_list(tests[1]),
            self.create_file_list(tests[2])
        ]

    @staticmethod
    def create_file_list(test: Path):
        return sorted([f for f in test.iterdir()])

    @staticmethod
    def get_test_run_time(files, test_num):
        if test_num == 0:
            return len(files[:44]) * 5 + len(files[44:]) * 10
        else:
            return len(files) * 10

    @staticmethod 
    def get_time_until_failure(total_test_time, file_idx, test_run):
        if test_run == 0:
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
    def get_data_frame_from_text(file, test_num):
        if test_num == 0:
            data_np = np.loadtxt(file)
            data_np = np.column_stack([
                (data_np[:, 0] + data_np[:, 1]) / 2,
                (data_np[:, 2] + data_np[:, 3]) / 2,
                (data_np[:, 4] + data_np[:, 5]) / 2,
                (data_np[:, 6] + data_np[:, 7]) / 2
            ])

            data = pd.DataFrame(
                data_np,
                columns=[
                'Bearing_One',
                'Bearing_Two',
                'Bearing_Three',
                'Bearing_Four'
                ]
            )
        else:
            data_np = np.loadtxt(file)
            data = pd.DataFrame(
                data_np,
                columns=[
                'Bearing_One',
                'Bearing_Two',
                'Bearing_Three',
                'Bearing_Four'
                ]
            )
        
        return data
    
    def get_windows(self, arr, window_size, stride, type="features"):
        windows = []
        for i in range(0, len(arr) - window_size, stride):
            if type == "features":
                window = arr[i:i+window_size]
                windows.append(window)
            else:
                window = arr[i + window_size - 1]
                windows.append(window)
        return windows 

    def get_samples(self, df, n_linspace=1000, dtype='int'):
        indices = np.linspace(0, len(df)-1, n_linspace, dtype=int)
        samples = df.iloc[indices, :]
        return samples
    
    def build_scaler(self, n_linspace=1000, scalar_type="standard"):
        all_data_for_scaling = []
        
        for test_num, test in enumerate(self.tests):
            for file in test:       
                df = self.get_data_frame_from_text(file, test_num)
                samples = self.get_samples(df, n_linspace)    
                all_data_for_scaling.extend(samples.values)

            print(f"Completed initial loading of test: {test_num}")

        all_data_array = np.array(all_data_for_scaling)
        scaler = StandardScaler()
        scaler.fit(all_data_array)
        print(f"Fitted scaler on {len(all_data_for_scaling)}")

        return scaler
    
    def build_normalized_data_frames(self, n_linspace=100):
        tests = []
        scaler = self.build_scaler()
            
        for test_num, test in enumerate(self.tests):
            all_test_data = []          
            total_test_time = self.get_test_run_time(test, test_num)
            
            for idx, file in enumerate(test):       
                df = self.get_data_frame_from_text(file, test_num)
                samples = self.get_samples(df)
                
                time_until_failure = self.get_time_until_failure(
                    total_test_time, 
                    idx, 
                    test_num
                )
                
                degradation = self.get_degradation(
                    time_until_failure, 
                    total_test_time
                )

                normalized_bearing_data = scaler.transform(samples.values)

                for sample in normalized_bearing_data:
                    file_data = pd.Series(
                        data=sample,
                        index=['Bearing_One', 'Bearing_Two', 'Bearing_Three', 'Bearing_Four']
                    )
                    file_data['time_until_failure'] = time_until_failure
                    file_data['degradation'] = degradation
                    all_test_data.append(file_data)

            print(f"completed building secondary data load {test_num}")
            result_df = pd.DataFrame(all_test_data).reset_index(drop=True)
            tests.append(result_df)
            
        return tests, scaler
    
    def run_pipeline(self):
        tests, scaler = self.build_normalized_data_frames()
        return tests, scaler