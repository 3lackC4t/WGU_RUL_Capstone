from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    # File Paths
    FILE_PATH = Path(__file__).parent.absolute()
    MODEL_DATA_PATH = FILE_PATH / "models" / "model_data"
    BI_LSTM_PATH = MODEL_DATA_PATH / "bi_lstm.keras"
    AUTO_ENCODER_PATH = MODEL_DATA_PATH / "autoencoder.keras"
    ENCODER_PATH = MODEL_DATA_PATH / "encoder.keras"
    DATA_PATH = FILE_PATH / "bearing_data"
    FILE_DATA_PATH = DATA_PATH / "file_data"
    ALLOWED_EXTENSIONS = {'txt', 'csv'}
    TEST_PATHS = [
        DATA_PATH / "1st_test" / "1st_test",
        DATA_PATH / "2nd_test" / "2nd_test",
        DATA_PATH / "3rd_test" / "4th_test" / "txt"
    ]

app = Flask(__name__)
app_settings = Settings()
app.config['UPLOAD_FOLDER'] = app_settings.FILE_DATA_PATH


def allowed_file(filename: str):
    split_filename = filename.split('.')
    if len(split_filename) > 2:
        # double file extension exploit
        return False

    extension = split_filename[1].lower()
    return extension in app_settings.ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def dashboard():
    return render_template("data_input.html")


@app.route('/api/input', methods=['GET', 'POST'])
def feed_input_data() -> float:
    if request.method == 'POST':
        life_span = int(request.form.get("vib-life-span"))
        bearing_rpm = int(request.form.get('bearing-rpm'))

        if 'vib-file' not in request.files:
            print("No file part")
            return render_template('error.html')

        file = request.files['vib-file']

        if file.filename == '':
            print('No selected file')
            return render_template('error.html')

        if file:
            print("The file was found, creating filename and saving")
            filename = secure_filename(file.filename)
            print(filename)
            file.save((app_settings.FILE_DATA_PATH / filename).as_posix())
            print('file was saved')

            try:
                with open(app_settings.FILE_DATA_PATH / filename) as f:
                    if filename.split('.')[-1].lower() == 'csv':
                        cleaned_data = PREPROCESSOR.get_cleaned_input(f, 'csv', bearing_rpm)
                    else:
                        cleaned_data = PREPROCESSOR.get_cleaned_input(f, 'txt', bearing_rpm)

                sensors = {}
                for idx, sensor in enumerate(cleaned_data):
                    print(f"Sensor Type: {type(sensor)}")
                    print(f"Sensor Shape: {sensor.shape}")
                    encoder_predictions = ENCODER.predict(sensor)
                    reshaped_predictions = encoder_predictions.reshape(encoder_predictions.shape[0], 1, encoder_predictions.shape[1])
                    health_prediction_raw = BI_LSTM.predict(reshaped_predictions)
                    mean_prediction = float(health_prediction_raw.mean())
                    sensors[f"sensor_{idx}"] = mean_prediction * life_span

                return jsonify({
                    'message': "Prediction returned succesfully",
                    'sensor_data': sensors
                }), 201
            except IOError:
                pass
            return "Unable to read file", 500


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()