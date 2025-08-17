from models.autoencoder import AutoEncoder
from models.preprocessor import Preprocessor

import numpy as np


class HealthPredictor:
    def __init__(self, model_file_path, reference_data_path=None, initial_training=True, test_paths=None):
        self.auto_encoder = AutoEncoder()
        self.preprocessor = Preprocessor(scalar='min_max')
        self.model_file_path = model_file_path
        self.reference_data_path = reference_data_path

        if not initial_training and not reference_data_path.exists():
            raise ValueError(f"No reference data was provided. If loading model from existing file reference data must be provided to create error threshold")

        self.baseline_error = None
        self.error_threshold = None

        if initial_training:
            # If this is the initial_training, then the build method will be run
            if test_paths:
                self.run_initial_training_build(test_paths)
            else:
                raise ValueError(f"test_paths is invalid or None type, please supply test paths for training data")
        else:
            # Model is already trained, load model from memory and return it.
            self.load_and_recalculate_baseline(model_file_path, reference_data_path)

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
            print(f"Test run time: {test_run_time:.2f}")

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
        X_train = np.array(complete_healthy_windows)
        X_test = np.array(complete_damaged_windows)

        # Save the reference data for future error_threshold calculation
        np.save(self.reference_data_path, X_train)

        # Train the autoencoder on the healthy data and extract the baseline error for the healthy data,
        # this is the threshold for a "healthy" bearing, deviation from this is the degree to which the bearing
        # is degraded
        self.auto_encoder.train_model(X_train, X_test)
        self.auto_encoder.save_model(self.model_file_path)

        self.baseline_error = self.auto_encoder.get_errors(X_train)
        self.error_threshold = np.percentile(self.baseline_error, 95)

    def load_and_recalculate_baseline(self, model_file_path, reference_data_path):
        # Load the model from .keras file, set the model status to trained 
        self.auto_encoder.load_model(model_file_path)
        self.auto_encoder.is_trained = True

        # load the reference data
        reference_data = np.load(reference_data_path)

        self.baseline_error = self.auto_encoder.get_errors(reference_data)
        self.error_threshold = np.percentile(self.baseline_error, 95)

        print(f"Loaded model from {str(model_file_path)} and calculated errors using data at {str(reference_data_path)} New error threshold is {self.error_threshold:.6f}")

    def get_mean_squared_error(self, predicted_errors) -> float:
        return float(np.mean(predicted_errors))
    
    def get_status(self, error):
        if self.error_threshold is None:
            return "Unknown", 0.0
        
        health_score = max(0.0, min(10, 1 - (error / (self.error_threshold * 3))))

        if health_score > 0.8:
            status = "healthy"
        elif health_score > 0.5:
            status = "moderate_wear"
        elif health_score > 0.3:
            status = "significant_wear"
        else:
            status = "Critical"

        return status, health_score

    def handle_input_data(self, data_file) -> dict[str:any]:
        # Load raw input data 
        raw_input_data = np.loadtxt(data_file)
        print(f"Raw input shape: {raw_input_data.shape}")

        all_sensor_data = {}
    
        for sensor_idx in range(raw_input_data.shape[1]):

            # create numpy array of sensor column
            sensor_data = raw_input_data[:, sensor_idx]
            print(f"Sensor data shape: {sensor_data.shape}")

            # Window the data
            input_data_windows = np.array(self.preprocessor.get_windows(sensor_data))
            print(f"Windowed data shape: {input_data_windows.shape}")

            if len(input_data_windows) == 0:
                all_sensor_data[f'sensor_{sensor_idx + 1}'] = {
                    'status': 'insufficient_data',
                    'health_score': 0.0,
                    'sensor_mse': 0.0,
                    'num_windows': 0
                } 

            # use the autoencoder to get the error from the data
            predicted_errors = self.auto_encoder.get_errors(input_data_windows)

            # MSE and health_status
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

