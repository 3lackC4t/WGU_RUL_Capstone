from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dataclasses import dataclass
from pathlib import Path

from models.health_predictor import HealthPredictor

@dataclass
class Settings:
    # File Paths
    FILE_PATH = Path(__file__).parent.absolute()
    MODEL_DATA_PATH = FILE_PATH / "model_data"
    REFERENCE_DATA_PATH = MODEL_DATA_PATH / "reference_data.npy"
    AUTO_ENCODER_PATH = MODEL_DATA_PATH / "autoencoder.keras"
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
health_predictor = HealthPredictor(
    model_file_path=app_settings.AUTO_ENCODER_PATH,
    reference_data_path=app_settings.REFERENCE_DATA_PATH,
    initial_training=False if app_settings.AUTO_ENCODER_PATH.exists() else True,
    test_paths=app_settings.TEST_PATHS
)


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
                    health_status = health_predictor.handle_input_data(f)
                    return jsonify(health_status), 201

            except IOError:
                pass
            return "Unable to read file", 500


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()