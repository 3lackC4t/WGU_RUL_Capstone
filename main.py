import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.autoencoder import LSTM_AutoEncoder
from models.bi_lstm import Bi_LSTM
from models.preprocessor import SensorPreprocessor
from API.model_API import *
from settings import Settings


def check_if_init(settings: Settings):
    return (
        settings.AUTO_ENCODER_PATH.exists() and
        settings.BI_LSTM_PATH.exists() and
        settings.ENCODER_PATH.exists()
    )


def initial_build(settings: Settings, testing=False):
    paths = [
        settings.DATA_PATH / "1st_test" / "1st_test",
        settings.DATA_PATH / "2nd_test" / "2nd_test",
        settings.DATA_PATH / "3rd_test" / "4th_test" / "txt"
    ]

    sensor_preprocessor = SensorPreprocessor(paths)
    sensor_dict = sensor_preprocessor.test_data
    sensor_data = [sensor_dict[f'test_{i + 1}']['sensor_data'] for i in range(len(sensor_dict.keys()))]

    # Autoencoder goes here
    # Bi-LSTM goes here

    all_sensor_windows = []
    all_degradation_windows = []

    for test in sensor_data:
        sensor_column_names = [col for col in test.columns if col.startswith('sensor_')]
        degradation_values = test['Degradation'].values
        for sensor_col in sensor_column_names:
            sensor_values = test[sensor_col].values

            sensor_windows = sensor_preprocessor.get_feature_windows(sensor_values) 
            degradation_windows = sensor_preprocessor.get_target_windows(degradation_values)

            all_sensor_windows.extend(sensor_windows)
            all_degradation_windows.extend(degradation_windows)

    X_all = np.array(all_sensor_windows)
    y_all = np.array(all_degradation_windows)

def main():
    app_settings = Settings()
    if not check_if_init(app_settings):
        initial_build(app_settings)
        main()
    else:
        pre_processor = SensorPreprocessor()
        app.run(debug=True)

    # run API

    # Main loop is to get data 


if __name__ == "__main__":
    main()