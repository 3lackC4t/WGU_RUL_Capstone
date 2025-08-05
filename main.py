import numpy as np
from sklearn.model_selection import train_test_split
from models.autoencoder import LSTM_AutoEncoder
from models.bi_lstm import Bi_LSTM
from models.preprocessor import Preprocessor
from API.model_API import *
from settings import Settings
from time import sleep
from pathlib import Path
import threading
from dataclasses import dataclass


def get_features(data):
    pass

def predict_on_data(features):
    # take in features
    # use preprocessor to turn into windows
    # take required windows, predict on it
    # Return it
    # wait 1 second
    # repeat 
    pass

def check_if_model_present(settings: Settings):
    return (settings.AUTO_ENCODER_PATH).exists(), (settings.BI_LSTM_PATH).exists()

def initial_build(settings, testing=False):
    autoencoder_present, lstm_present = check_if_model_present(settings)
    if not autoencoder_present or lstm_present:
        paths = [
            settings.DATA_PATH / "1st_test" / "1st_test",
            settings.DATA_PATH / "2nd_test" / "2nd_test",
            settings.DATA_PATH / "3rd_test" / "4th_test" / "txt"
        ]
        preprocessor = Preprocessor(paths)
        tests, scaler = preprocessor.run_pipeline(testing)

        all_X, all_y = [], []

        for test in tests:
            X = preprocessor.get_windows(test, 16, 2, 'features')
            y = preprocessor.get_windows(test, 16, 2, 'targets')
            all_X.extend(X)
            all_y.extend(y)

        X, y = np.array(all_X), np.array(all_y)

        if not autoencoder_present:
            auto_encoder_builder = LSTM_AutoEncoder()
            auto_encoder_builder.fit_autoencoder(X, X)
            autoencoder, encoder = auto_encoder_builder.autoencoder, auto_encoder_builder.encoder

        features = encoder.predict(X)
        windowed_features = preprocessor.get_windows(features, 16, 2, 'features')
        windowed_targets = preprocessor.get_windows(y, 16, 2, 'targets')

        X_train, X_test, y_train, y_test = train_test_split(
            windowed_features, windowed_targets,
            test_size=0.2, shuffle=True
        )

        if not lstm_present:
            bi_lstm_builder = Bi_LSTM()
            bi_lstm_builder.fit_bi_lstm(X_train, y_train, X_test, y_test)
            bi_lstm = bi_lstm_builder.model

        return preprocessor, bi_lstm, autoencoder, encoder

    else:
        preprocessor = Preprocessor(None)
        bi_lstm_builder = Bi_LSTM()
        auto_encoder_builder = LSTM_AutoEncoder()
        return preprocessor, bi_lstm_builder.model, auto_encoder_builder.autoencoder, auto_encoder_builder.encoder

def main():
    app_settings = Settings()
    preprocessor, bi_lstm, autoencoder, encoder = initial_build(settings=app_settings, testing=True)
    # run API

    # Main loop is to get data 


if __name__ == "__main__":
    main()