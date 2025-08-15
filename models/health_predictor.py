from autoencoder import AutoEncoder
from preprocessor import Preprocessor

import numpy as np


class HealthPredictor:
    def __init__(self, initial_training=True, test_paths=None):
        self.auto_encoder = AutoEncoder()
        self.preprocessor = Preprocessor()
        self.baseline_error = None
        self.error_threshold = None

        if initial_training:
            if test_paths:
                self.run_inital_training_build(test_paths)
            else:
                raise ValueError(f"test_paths is invalid or None type, please supply test paths for training data")
        else:
            # Model is already trained, load model from memory and return it.
            pass

    def run_initial_training_build(self, test_paths: list) -> None:
        complete_healthy_windows = []
        complete_damaged_windows = []

        for test_idx, test_path in enumerate(test_paths):
            print(f"\nProcessing test {test_idx + 1}: {test_path}")

            # Get complete file list form test
            files = self.preprocessor.create_file_list(test_path)
            print(f"Found {len(files)} files for test {test_idx + 1}")

            # Calculate total time in seconds of the run
            test_run_time = self.preprocessor.get_test_run_time(files, test_idx)
            print(f"Test run time: {test_run_time:..2}")

            # using preprocessors proportion, get the sample file indices that will be used
            indices = self.preprocessor.get_indices(files)
            print(f"Number of samples: {len(indices)}")

            # Use the base pipeline to create the healthy and damaged windows for this test
            test_healthy_windows, test_damaged_windows = self.preprocessor.bearing_test_data_pipeline(
                files, test_run_time, indices, test_idx
            )

            # move the healthy and damaged windows into their respective arrays
            complete_healthy_windows.extend(test_healthy_windows)
            complete_damaged_windows.extend(test_damaged_windows)



        # Create complete, universal, 1D healthy sensor data
        # training data should be much larger than test_data
        training_data = np.array(complete_healthy_windows)
        test_data = np.array(complete_damaged_windows)

        # Normalize training and test data
        X_train = self.preprocessor.get_scaled_data(training_data)
        X_test = self.preprocessor.get_scaled_data(test_data)

        # Train the autoencoder on the healthy data and extract the baseline error for the healthy data,
        # this is the threshold for a "healthy" bearing, deviation from this is the degree to which the bearing
        # is degraded
        self.auto_encoder.train_model(X_train, X_test)
        self.baseline_error = self.auto_encoder.get_error(X_train)

    def get_mean_squared_error(self, predicted_errors) -> float:
        return float(np.mean(predicted_errors))
    
    def get_status(self)

    def handle_input_data(self, data_file) -> dict[str:any]:
        # Load raw input data 
        raw_input_data = np.loadtxt(data_file)

        all_sensor_data = {}
    
        for sensor_idx, col in enumerate(raw_input_data.columns):

            # create numpy array of sensor column
            sensor_data = raw_input_data[:, col]

            # Window the data
            input_data_windows = self.preprocessor.get_windows(sensor_data)

            # use the autoencoder to get the error from the data
            predicted_errors = self.auto_encoder.get_errors(input_data_windows)

            # MSE
            sensor_mse = self.get_mean_squared_error(predicted_errors)
            status, health_score = self.get_status(sensor_mse)

            # Create JSON-able return dictionary
            all_sensor_data[f'sensor_{sensor_idx}'] = {
                'status': status,
                'health_score': health_score,
                'sensor_mse': sensor_mse,
                'num_windows': len(input_data_windows),
                'error_statistics': {
                    'mean': sensor_mse,
                    'std': float(np.std(predicted_errors)),
                    'max': float(np.max(predicted_errors)),
                    'min': float(np.min(predicted_errors))
                }
            }

            return all_sensor_data

