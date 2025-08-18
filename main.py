from pathlib import Path

from models.bearing_preprocessor import NASABearingPreprocessor

BASE_PATH = Path(__file__).parent.absolute()
BEARING_DATA_PATH = BASE_PATH / "bearing_data"
MODEL_DATA_PATH = BASE_PATH / "model_data"
AUTOENCODER_PATH = MODEL_DATA_PATH / "autoencoder.keras"
REFERENCE_DATA_PATH = MODEL_DATA_PATH / "reference_data.npy"


test_paths = [
    (BEARING_DATA_PATH / "1st_test" / "1st_test", 1),
    (BEARING_DATA_PATH / "2nd_test" / "2nd_test", 2),
    (BEARING_DATA_PATH / "3rd_test" / "4th_test" / "txt", 3)
]


def main() -> None:
    
    print("Initializing preprocessor")
    preprocessor = NASABearingPreprocessor(
        window_size=2048,
        overlap=0.5,
        scaler_type='standard',
        health_threshold=0.5,
        extract_features=True
    )

    
    healthy_data, degraded_data, metadata = preprocessor.process_multiple_tests(
        test_paths,
        sample_proportion=0.1
    )
    print("="*60)
    print(f"\nHealthy Data Shape: {healthy_data.shape}\nDegraded Data Shape: {degraded_data.shape}\n")
    print("="*60)


    if healthy_data.shape[1] > 0:
        healthy_scaled = preprocessor.fit_transform(healthy_data)
        print(f"\nScaled healthy data shape: {healthy_scaled.shape}")
        print(f"Feature dimension: {preprocessor.get_feature_dim()}")

    if degraded_data.shape[1] > 0:
        degraded_scaled = preprocessor.transform(degraded_data)
        print(f"Scaled degraded data shape: {degraded_scaled.shape}")


if __name__ == "__main__":
    main() 