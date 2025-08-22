import numpy as np
from pathlib import Path
from typing import Dict
from flask import Flask, jsonify, redirect, request, render_template
from werkzeug.utils import secure_filename
from models.bearing_preprocessor import NASABearingPreprocessor
from models.autoencoder import Autoencoder
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent.absolute()
BEARING_DATA_PATH = BASE_PATH / "bearing_data"
MODEL_DATA_PATH = BASE_PATH / "model_data"
UPLOAD_FOLDER = BEARING_DATA_PATH / "file_data"
AUTOENCODER_PATH = MODEL_DATA_PATH / "autoencoder.keras"
REFERENCE_DATA_PATH = MODEL_DATA_PATH / "reference_data.npy"
PREPROCESSOR_PATH = MODEL_DATA_PATH / 'preprocessor.pkl'

HEALTHY_TEST_DATA = BEARING_DATA_PATH / "1st_test" / "1st_test" / "2003.10.22.12.06.24"
DEGRADED_TEST_DATA = BEARING_DATA_PATH / "1st_test" / "1st_test" / "2003.11.24.20.47.32" 


test_paths = [
    (BEARING_DATA_PATH / "1st_test" / "1st_test", 1),
    (BEARING_DATA_PATH / "2nd_test" / "2nd_test", 2),
    (BEARING_DATA_PATH / "3rd_test" / "4th_test" / "txt", 3)
]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER.as_posix()

def model_init() -> None:

    print("Initializing preprocessor")
    preprocessor = NASABearingPreprocessor(
        window_size=2048,
        overlap=0.5,
        scaler_type='standard',
        health_threshold=0.7,
        proportion=0.3,
        extract_features=True
    )

    reference_data = None

    models_exits = AUTOENCODER_PATH.exists() and PREPROCESSOR_PATH.exists()

    if models_exits:
        preprocessor = NASABearingPreprocessor()
        preprocessor.load_preprocessor(PREPROCESSOR_PATH)

        input_dim = preprocessor.get_feature_dim()
        autoencoder = Autoencoder(
            input_dim=input_dim,
            epochs=100,
            batch_size=32
        )

        autoencoder.load_model(AUTOENCODER_PATH)

        reference_data = None
        if REFERENCE_DATA_PATH.exists():
            preprocessor.load_data(REFERENCE_DATA_PATH)
            reference_data = preprocessor.reference_threshold

        print("Models loaded Succesfully")

        return autoencoder, preprocessor

    else:
        print("Training new models...")

        
        healthy_data, metadata = preprocessor.process_multiple_tests(
            test_paths
        )
        print("="*60)
        print(f"\nHealthy Data Shape: {healthy_data.shape}\n")
        print("="*60)

        if healthy_data.shape[1] > 0:
            healthy_scaled = preprocessor.fit_transform(healthy_data)
            print(f"\nScaled healthy data shape: {healthy_scaled.shape}")
            print(f"Feature dimension: {preprocessor.get_feature_dim()}")

            print(f"Creating Autoencoder")
            autoencoder = Autoencoder(
                input_dim=healthy_scaled.shape[1],
                epochs=100,
                batch_size=32
            ) 

            autoencoder.build_model()
            model_history = autoencoder.fit(healthy_scaled)

            print(f"Saving autoencoder to {AUTOENCODER_PATH.as_posix()}")
            MODEL_DATA_PATH.mkdir(parents=True, exist_ok=True)
            autoencoder.save_model(AUTOENCODER_PATH)

            print(f"Calculating reference data and threshold")
            reference_data = autoencoder.predict_on_input(healthy_scaled)
            reference_mse = preprocessor.get_error_mse(healthy_scaled, reference_data)
            preprocessor.calculate_threshold(reference_mse)
            preprocessor.save_preprocessor(PREPROCESSOR_PATH)

            preprocessor.save_data(REFERENCE_DATA_PATH, reference_mse)

            print("Models saved succesfully!")

            return autoencoder, preprocessor


def handle_input(input_file: Path) -> Dict['str', float]:
    print("Calculating bearing health")
    processed_file = preprocessor.process_input_file(input_file)

    result = {}

    print("File processed, parsing sensor data")
    for bearing_name in processed_file.keys():
        result[bearing_name] = {}

        processed_data = processed_file[bearing_name]
        reconstruction = autoencoder.predict_on_input(processed_data)

        mse = preprocessor.get_error_mse(processed_data, reconstruction)
        result[bearing_name]['mse_raw'] = list(mse)
        result[bearing_name]['mse_mean'] = float(np.mean(mse))

        if preprocessor.reference_threshold is not None:
            # Detect anomalies
            anomalies = mse > preprocessor.reference_threshold
            rate = float(np.mean(anomalies) * 100)
            result[bearing_name]['anomaly_rate'] = rate
            print(f"    Anomaly rate: {rate:.1f}%")

            # Health score 
            health_score = preprocessor.calculate_health_score(mse)
            result[bearing_name]['health_score'] = float(np.mean(health_score))
            print(f"    Average health: {np.mean(health_score):.1f}%")

            health_status = get_bearing_status(mse, preprocessor)
            result[bearing_name]['health_status'] = health_status

    for bearing, bearing_data in result.items():
        print(bearing)
        for metric, value in bearing_data.items():
            print(f"{metric}: {type(value)}")

    return result

def get_bearing_status(mse: np.ndarray, preprocesser: NASABearingPreprocessor) -> str:
    mean_mse = np.mean(mse)

    if mean_mse < preprocesser.reference_mean + preprocesser.reference_std:
        return 'Healthy'
    elif mean_mse < preprocesser.reference_mean + 2 * preprocesser.reference_std:
        return 'Warning'
    elif mean_mse < preprocesser.reference_threshold:
        return 'Degraded'
    else:
        return 'Critical'


autoencoder, preprocessor = model_init()


@app.route('/', methods=['GET', 'POST'])
def data_input():
    return render_template('data_input.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/api/input', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        print("POST Detected")
        if 'vib-file' not in request.files:
            print("File was not found")
            return redirect(request.url)
        
        file = request.files['vib-file']

        if file.filename == '':
            print("Filename not found")
            return redirect(request.url)
        
        if file:
            print("File found, attempting to predict")
            filename = secure_filename(file.filename)
            file.save(Path(app.config['UPLOAD_FOLDER'], filename))

            error_data = handle_input(UPLOAD_FOLDER / filename)

            return jsonify(error_data)


def main() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    main() 