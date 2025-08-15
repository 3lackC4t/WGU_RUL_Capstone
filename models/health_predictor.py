from autoencoder import AutoEncoder
from preprocessor import Preprocessor

import numpy as np


class HealthPredictor:
    def __init__(self, initial_training=True, test_paths=None):
        self.auto_encoder = AutoEncoder()
        self.preprocessor = Preprocessor()
        self.base_line_error = None

        if initial_training:
            if test_paths:
                self.run_inital_training_build(test_paths)
            else:
                raise ValueError(f"test_paths is invalid or None type, please supply test paths for training data")
        else:
            # Model is already trained, load model from memory and return it.
            pass

    def get_status(self, error) -> tuple[str, float]:
        pass

    def run_initial_training_build(self, test_paths: list) -> None:
        complete_healthy_windows = []
        complete_damaged_windows = []

        for test_idx, test in enumerate(test_paths):

            # Get complete file list form test
            files = self.preprocessor.create_file_list(test)

            # Calculate total time in seconds of the run
            test_run_time = self.preprocessor.get_test_run_time(files, test_idx)

            # using preprocessors proportion, get the sample file indices that will be used
            indices = self.preprocessor.get_indices(files)

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

        # Train the autoencoder on the healthy data and extract the baseline error for the healthy data,
        # this is the threshold for a "healthy" bearing, deviation from this is the degree to which the bearing
        # is degraded
        self.auto_encoder.train_model(training_data, test_data)
        self.base_line_error = self.auto_encoder.get_error(training_data)

    def get_mean_squared_error(self, predicted_error) -> float:
        pass

    def handle_input_data(self, data_file) -> dict[str:any]:
        # Load raw input data 
        raw_input_data = np.loadtxt(data_file)

        all_sensor_data = {}
    
        for sensor_idx, col in enumerate(raw_input_data.columns):

            # create numpy array of sensor column
            sensor_data = raw_input_data[:, col]

            # Window the data
            input_data_windows = self.preprocessor.get_input_windows(sensor_data)

            # use the autoencoder to get the error from the data
            predicted_error = self.auto_encoder.get_error(input_data_windows)

            # MSE
            sensor_mse = self.get_mean_squared_error(predicted_error)
            status = self.get_status(sensor_mse)

            # Create JSON-able return dictionary
            all_sensor_data[f'sensor_{sensor_idx}'] = {
                'status': status,
                'sensor_mse': sensor_mse
            }

